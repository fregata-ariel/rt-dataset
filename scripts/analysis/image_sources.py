"""Image-source (virtual-source) analysis of an RF-camera street-canyon scene.

This exploratory demo traces a specular/refraction multipath channel in the
built-in ``simple_street_canyon`` scene, reconstructs the virtual (image)
source of every path from the UE-local arrival direction and the path delay,
and checks that the virtual source explains both the path geometry and the
measured angular image.

The premise is that a specular multipath component behaves like a point source
at its image source, which is the basis for representing multipath as Gaussians
placed at image-source positions.

Outputs (under ``--out``):
  topview.png            building footprints, BS, UEs and virtual sources
  ue_XX_image.png        front-hemisphere direction-cosine image with VS markers
  virtual_sources.csv    one row per traced path
  report.md              setup, per-UE tables, residuals and caveats

Usage:
  python scripts/analysis/image_sources.py --out data/generated/analysis/image_sources
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sionna.rt import Receiver, Transmitter, load_scene
from sionna.rt.constants import InteractionType
from sionna.rt.scene import simple_street_canyon

from plateau_rt.adapters.plotting.rf_camera_plots import (
    Marker,
    image_extent,
    normalized_power_db,
    pyplot,
    save_direction_image,
)
from plateau_rt.adapters.sionna.rf_patterns import HEMISPHERE_SPLIT_PATTERN
from plateau_rt.adapters.sionna.rf_tracing import (
    aperture_cfrs,
    configure_rf_camera_arrays,
    trace_paths,
)
from plateau_rt.domain.rf_camera.calibration import calibrate_angular_cfr
from plateau_rt.domain.rf_camera.camera import (
    HEMISPHERES,
    look_at_orientation,
    to_solid_angle_amplitude,
)
from plateau_rt.domain.rf_camera.delay import (
    SPEED_OF_LIGHT_M_S,
    angular_cfr_to_delay,
    circular_delay_error_s,
    propagating_direction_mask,
)
from plateau_rt.domain.rf_camera.image_sources import (
    arrival_unit_vectors,
    find_image_peaks,
    hann_taper,
    local_direction_to_image_coords,
    match_peaks_to_sources,
    pattern_summed_element_power,
    point_to_ray_distance,
    polyline_length,
    source_recall,
    unfold_specular_chain,
    virtual_source_positions,
    world_to_local_directions,
)
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft, frequency_offsets

CARRIER_FREQUENCY_HZ = 3.5e9
BANDWIDTH_HZ = 100e6
NUM_FREQUENCY_BINS = 64

# Interactions that unfold to a valid image source. Transmission is modelled by
# Sionna as straight-through (``ko_local_trans = ki_local``), so refraction
# chains still unfold; diffuse and diffraction do not.
IMAGE_SOURCE_INTERACTIONS = (
    InteractionType.NONE,
    InteractionType.SPECULAR,
    InteractionType.REFRACTION,
)

# Building bounding boxes read from scene.objects[name].mi_mesh.bbox() (printed
# at run time). The street canyon runs along x; the free street band is roughly
# y in (-8.6, 9.6), so all UEs sit inside that band away from any building.
# The BS sits at the west street end, about 20 m up, so facade reflections and
# their virtual sources fall in the UEs' front hemisphere.
BS_POSITION = (-85.0, 0.0, 20.0)
BS_LOOK_AT = (0.0, 0.0, 8.0)

# Each UE is an RF camera at 1.5 m, looking west down the street toward the BS
# (chosen so LoS and both facade reflections fall inside the front hemisphere).
LOOK_AT = (-85.0, 0.0, 8.0)
UE_VIEWS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("ue_00", (-25.0, 0.0, 1.5), LOOK_AT),
    ("ue_01", (-5.0, 0.0, 1.5), LOOK_AT),
    ("ue_02", (15.0, 0.0, 1.5), LOOK_AT),
    ("ue_03", (35.0, 0.0, 1.5), LOOK_AT),
]


@dataclass
class PathRecord:
    """Per-path geometry, virtual source and image-check results."""

    ue: str
    path_index: int
    bounce: int
    has_refraction: bool
    is_image_source: bool
    interaction_chain: list[int]
    power_linear: float
    power_db: float
    tau_s: float
    vs: np.ndarray
    in_front: bool
    ky: float
    kz: float
    kx: float
    image_power_db: float
    check1_residual_m: float
    check2_residual_m: float
    ray_distance_m: float
    check3_unfold_m: float


@dataclass
class ViewResult:
    """Analysis of one UE: paths, developed image and peak matches."""

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    records: list[PathRecord]
    peaks: list[tuple[int, int]]
    peak_kykz: np.ndarray
    image: np.ndarray
    center_power: np.ndarray
    ky_over_k: np.ndarray
    kz_over_k: np.ndarray
    valid_mask: np.ndarray
    delay_s: np.ndarray
    power_volume: np.ndarray
    peak_match_path: np.ndarray
    peak_match_distance: np.ndarray
    strong_sources: list[PathRecord] = field(default_factory=list)
    strong_recalled: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    strong_resolvable: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    @property
    def matched_peaks(self) -> int:
        """Number of image peaks matched to a candidate within the gate."""
        return int(np.sum(self.peak_match_path >= 0))

    @property
    def precision(self) -> float:
        """Fraction of peaks matched to a candidate source."""
        total = len(self.peaks)
        return self.matched_peaks / total if total else float("nan")

    @property
    def recalled_count(self) -> int:
        """Number of strong sources with a peak inside the beamwidth."""
        return int(np.sum(self.strong_recalled))

    @property
    def strong_source_count(self) -> int:
        """Number of strong front-hemisphere image sources."""
        return len(self.strong_sources)

    @property
    def strong_recall(self) -> float:
        """Fraction of strong sources recalled by the image."""
        total = self.strong_source_count
        return self.recalled_count / total if total else float("nan")

    @property
    def resolvable_count(self) -> int:
        """Number of strong sources separated by at least one beamwidth."""
        return int(np.sum(self.strong_resolvable))

    @property
    def resolvable_recalled(self) -> int:
        """Number of resolvable strong sources recalled by the image."""
        return int(np.sum(self.strong_recalled & self.strong_resolvable))

    @property
    def resolvable_recall(self) -> float:
        """Fraction of resolvable strong sources recalled by the image."""
        total = self.resolvable_count
        return self.resolvable_recalled / total if total else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated/analysis/image_sources"),
        help="output directory",
    )
    parser.add_argument("--carrier-hz", type=float, default=CARRIER_FREQUENCY_HZ)
    parser.add_argument("--bandwidth-hz", type=float, default=BANDWIDTH_HZ)
    parser.add_argument("--num-frequency-bins", type=int, default=NUM_FREQUENCY_BINS)
    parser.add_argument("--rx-rows", type=int, default=16)
    parser.add_argument("--rx-cols", type=int, default=16)
    parser.add_argument("--spacing", type=float, default=0.5, help="aperture spacing in lambda")
    parser.add_argument("--fft-rows", type=int, default=128)
    parser.add_argument("--fft-cols", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-threshold-db", type=float, default=-20.0)
    parser.add_argument("--peak-size", type=int, default=5)
    parser.add_argument(
        "--taper",
        choices=("hann", "none"),
        default="hann",
        help="aperture taper applied before the angular FFT",
    )
    return parser.parse_args()


def building_boxes(
    scene: object,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Return axis-aligned x-y footprints of the scene's volumetric objects."""
    boxes = []
    for name in scene.objects:
        bbox = scene.objects[name].mi_mesh.bbox()
        low = np.array([bbox.min[0], bbox.min[1], bbox.min[2]], dtype=float)
        high = np.array([bbox.max[0], bbox.max[1], bbox.max[2]], dtype=float)
        print(f"  {name:12s} min={low.round(3).tolist()} max={high.round(3).tolist()}")
        if high[2] - low[2] > 1.0:
            boxes.append((name, tuple(low.tolist()), tuple(high.tolist())))
    return boxes


def analyze_view(
    *,
    name: str,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float],
    rx_index: int,
    aperture: np.ndarray,
    tau: np.ndarray,
    theta_r: np.ndarray,
    phi_r: np.ndarray,
    valid: np.ndarray,
    vertices: np.ndarray,
    interactions: np.ndarray,
    path_power: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    args: argparse.Namespace,
) -> ViewResult:
    """Reconstruct virtual sources, develop the front image and match peaks."""
    num_paths = tau.shape[-1]
    non_none = interactions != InteractionType.NONE
    bounce_counts = np.sum(interactions == InteractionType.SPECULAR, axis=0)
    has_refraction = np.any(interactions == InteractionType.REFRACTION, axis=0)

    r_hat = arrival_unit_vectors(theta_r, phi_r)
    virtual_sources = virtual_source_positions(position, tau, theta_r, phi_r)
    local = world_to_local_directions(r_hat, orientation)
    ky, kz, in_front = local_direction_to_image_coords(local)

    ue = np.asarray(position, dtype=float)
    bs = np.asarray(BS_POSITION, dtype=float)
    vertex_axis = vertices[:, rx_index, 0, :, :]

    records: list[PathRecord] = []
    for path in range(num_paths):
        if not bool(valid[rx_index, 0, path]):
            continue
        mask = non_none[:, rx_index, 0, path]
        points = vertex_axis[mask, path]
        chain = interactions[:, rx_index, 0, path][mask]
        interaction_chain = interactions[:, rx_index, 0, path].tolist()
        is_image_source = bool(np.all(np.isin(interaction_chain, IMAGE_SOURCE_INTERACTIONS)))
        polyline = np.vstack([bs[None, :], points, ue[None, :]])
        check1 = polyline_length(polyline) - SPEED_OF_LIGHT_M_S * float(tau[rx_index, 0, path])

        check2 = np.nan
        if int(bounce_counts[rx_index, 0, path]) == 1 and not bool(
            has_refraction[rx_index, 0, path]
        ):
            v1 = points[0]
            check2 = abs(
                float(
                    np.linalg.norm(v1 - bs)
                    - np.linalg.norm(v1 - virtual_sources[rx_index, 0, path])
                )
            )

        anchor = points[-1] if points.shape[0] else bs
        ray_distance = point_to_ray_distance(anchor, ue, r_hat[rx_index, 0, path])

        check3 = np.nan
        if is_image_source:
            chain_is_reflection = chain == InteractionType.SPECULAR
            unfolded = unfold_specular_chain(bs, points, chain_is_reflection, ue)
            check3 = float(np.linalg.norm(unfolded - virtual_sources[rx_index, 0, path]))

        power_linear = float(path_power[rx_index, path])
        power_db = float(10.0 * np.log10(max(power_linear, 1e-30)))
        kx_value = float(local[rx_index, 0, path, 0])
        in_front_value = bool(in_front[rx_index, 0, path])
        # The image is A = kx*U, so a source's image power is its path power
        # times kx^2. Back-hemisphere sources do not appear in the front image.
        if in_front_value:
            image_power_db = power_db + 20.0 * np.log10(max(kx_value, 1e-6))
        else:
            image_power_db = float("nan")

        records.append(
            PathRecord(
                ue=name,
                path_index=path,
                bounce=int(bounce_counts[rx_index, 0, path]),
                has_refraction=bool(has_refraction[rx_index, 0, path]),
                is_image_source=is_image_source,
                interaction_chain=interaction_chain,
                power_linear=power_linear,
                power_db=power_db,
                tau_s=float(tau[rx_index, 0, path]),
                vs=virtual_sources[rx_index, 0, path],
                in_front=in_front_value,
                ky=float(ky[rx_index, 0, path]),
                kz=float(kz[rx_index, 0, path]),
                kx=kx_value,
                image_power_db=image_power_db,
                check1_residual_m=float(check1),
                check2_residual_m=float(check2),
                ray_distance_m=float(ray_distance),
                check3_unfold_m=float(check3),
            )
        )

    front = aperture[HEMISPHERES.index("front")]
    if args.taper == "hann":
        front = front * hann_taper(front.shape[0], front.shape[1])[:, :, None]
    calibration = calibrate_angular_cfr(
        aperture_to_angular_fft(front, fft_rows=args.fft_rows, fft_cols=args.fft_cols),
        aperture_rows=args.rx_rows,
        aperture_cols=args.rx_cols,
        horizontal_spacing_lambda=args.spacing,
        vertical_spacing_lambda=args.spacing,
    )
    image = to_solid_angle_amplitude(calibration.cfr, calibration.ky_over_k, calibration.kz_over_k)
    center_power = np.abs(image[:, :, args.num_frequency_bins // 2]) ** 2
    valid_mask = propagating_direction_mask(calibration.ky_over_k, calibration.kz_over_k)

    peaks = find_image_peaks(
        center_power,
        mask=valid_mask,
        threshold_db=args.peak_threshold_db,
        size=args.peak_size,
    )
    peak_kykz = np.array(
        [[calibration.ky_over_k[col], calibration.kz_over_k[row]] for row, col in peaks],
        dtype=float,
    ).reshape(-1, 2)

    delay_volume = angular_cfr_to_delay(image, frequency_offsets_hz)
    power_volume = np.abs(delay_volume.cir) ** 2

    # Match against front-hemisphere image sources (LoS/specular/refraction
    # chains). Back-hemisphere sources do not contribute to the front image.
    beamwidth = 1.0 / (max(args.rx_rows, args.rx_cols) * args.spacing)
    # Rayleigh (first-null) separation of the tapered aperture: a Hann taper
    # doubles the main-lobe width relative to the untampered aperture.
    resolution = beamwidth * (2.0 if args.taper == "hann" else 1.0)
    source_kykz = np.array([[rec.ky, rec.kz] for rec in records], dtype=float).reshape(-1, 2)
    source_ids = np.array([rec.path_index for rec in records], dtype=int)
    candidate_mask = np.array([rec.in_front and rec.is_image_source for rec in records], dtype=bool)
    match_path, match_distance = match_peaks_to_sources(
        peak_kykz,
        source_kykz,
        source_ids=source_ids,
        candidate_mask=candidate_mask,
        max_distance=beamwidth,
    )

    # Recall of the strong front-hemisphere image sources. The image is
    # A = kx*U, so a source's image power is its path power times kx^2 and the
    # strong threshold is applied to that kx^2-weighted power so it matches the
    # peak threshold in the developed image.
    front_image = [rec for rec in records if rec.in_front and rec.is_image_source]
    if front_image:
        max_image_power_db = max(rec.image_power_db for rec in front_image)
        strong_sources = [
            rec
            for rec in front_image
            if rec.image_power_db >= max_image_power_db + args.peak_threshold_db
        ]
    else:
        strong_sources = []
    strong_kykz = np.array([[rec.ky, rec.kz] for rec in strong_sources], dtype=float).reshape(-1, 2)
    strong_recalled = source_recall(peak_kykz, strong_kykz, max_distance=beamwidth)
    if len(strong_sources) > 1:
        separations = np.linalg.norm(strong_kykz[:, None, :] - strong_kykz[None, :, :], axis=-1)
        np.fill_diagonal(separations, np.inf)
        strong_resolvable = np.min(separations, axis=1) >= resolution
    else:
        strong_resolvable = np.ones(strong_kykz.shape[0], dtype=bool)

    return ViewResult(
        name=name,
        position=position,
        orientation=orientation,
        records=records,
        peaks=peaks,
        peak_kykz=peak_kykz,
        image=image,
        center_power=center_power,
        ky_over_k=calibration.ky_over_k,
        kz_over_k=calibration.kz_over_k,
        valid_mask=valid_mask,
        delay_s=delay_volume.delay_s,
        power_volume=power_volume,
        peak_match_path=match_path,
        peak_match_distance=match_distance,
        strong_sources=strong_sources,
        strong_recalled=strong_recalled,
        strong_resolvable=strong_resolvable,
    )


def plot_topview(
    output_path: Path,
    boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    results: list[ViewResult],
) -> None:
    """Draw building footprints, BS/UEs and virtual sources."""
    plt = pyplot()
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 7))
    for label, low, high in boxes:
        ax.add_patch(
            Rectangle(
                (low[0], low[1]),
                high[0] - low[0],
                high[1] - low[1],
                facecolor="0.85",
                edgecolor="0.4",
                label=label,
            )
        )

    ax.scatter([BS_POSITION[0]], [BS_POSITION[1]], marker="*", s=280, c="red", label="BS", zorder=5)
    for result in results:
        ax.scatter(
            [result.position[0]],
            [result.position[1]],
            marker="s",
            s=70,
            c="black",
            label=result.name,
        )

    all_bounce = [rec.bounce for r in results for rec in r.records if rec.is_image_source]
    all_power = np.array(
        [rec.power_linear for r in results for rec in r.records if rec.is_image_source]
    )
    power_max = float(all_power.max()) if all_power.size else 1.0
    scatter = None
    for result in results:
        records = [rec for rec in result.records if rec.is_image_source]
        if not records:
            continue
        vs = np.array([rec.vs for rec in records])
        power = np.array([rec.power_linear for rec in records])
        bounce = [rec.bounce for rec in records]
        scatter = ax.scatter(
            vs[:, 0],
            vs[:, 1],
            c=bounce,
            cmap="viridis",
            s=25 + 250 * power / power_max,
            alpha=0.8,
            zorder=4,
        )
        for rec in records:
            ax.plot(
                [result.position[0], rec.vs[0]],
                [result.position[1], rec.vs[1]],
                color="0.6",
                linewidth=0.4,
                zorder=1,
            )

    if scatter is not None and all_bounce:
        cbar = fig.colorbar(scatter, ax=ax, label="specular bounce count")
        cbar.set_ticks(sorted(set(all_bounce)))
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title("Simple street canyon: BS, UEs and image sources (z-independent top view)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(output_path: Path, results: list[ViewResult]) -> None:
    """Write one row per traced valid path."""
    columns = [
        "ue",
        "path_index",
        "bounce",
        "has_refraction",
        "is_image_source",
        "interaction_chain",
        "power_db",
        "tau_ns",
        "vs_x",
        "vs_y",
        "vs_z",
        "in_front",
        "ky_over_k",
        "kz_over_k",
        "kx",
        "image_power_db",
        "check1_residual_m",
        "check2_residual_m",
        "ray_distance_m",
        "check3_unfold_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for result in results:
            for rec in result.records:
                writer.writerow(
                    [
                        rec.ue,
                        rec.path_index,
                        rec.bounce,
                        rec.has_refraction,
                        rec.is_image_source,
                        "-".join(str(i) for i in rec.interaction_chain),
                        f"{rec.power_db:.3f}",
                        f"{rec.tau_s * 1e9:.3f}",
                        f"{rec.vs[0]:.4f}",
                        f"{rec.vs[1]:.4f}",
                        f"{rec.vs[2]:.4f}",
                        rec.in_front,
                        f"{rec.ky:.6f}",
                        f"{rec.kz:.6f}",
                        f"{rec.kx:.6f}",
                        "nan" if np.isnan(rec.image_power_db) else f"{rec.image_power_db:.3f}",
                        f"{rec.check1_residual_m:.6f}",
                        "nan"
                        if np.isnan(rec.check2_residual_m)
                        else f"{rec.check2_residual_m:.6f}",
                        f"{rec.ray_distance_m:.6f}",
                        "nan" if np.isnan(rec.check3_unfold_m) else f"{rec.check3_unfold_m:.6f}",
                    ]
                )


def build_summary(results: list[ViewResult], args: argparse.Namespace) -> dict[str, object]:
    """Collect the numeric summary used by both stdout and report.md."""
    bounce_histogram: dict[int, int] = {}
    refraction_image_source_paths = 0
    non_image_source_paths = 0
    image_source_paths = 0
    check1: list[float] = []
    check2: list[float] = []
    ray_distances: list[float] = []
    check3: list[float] = []
    peak_distances: list[float] = []
    strong_peak_distances: list[float] = []
    delay_errors: list[float] = []
    peak_count = 0
    matched_peaks = 0
    unresolved = 0
    front_vs_count = 0
    strong_count = 0
    recalled_count = 0
    resolvable_count = 0
    resolvable_recalled = 0
    missed_strong: list[dict[str, object]] = []
    per_view: list[dict[str, object]] = []
    beamwidth = 1.0 / (max(args.rx_rows, args.rx_cols) * args.spacing)
    resolution = beamwidth * (2.0 if args.taper == "hann" else 1.0)
    period = args.num_frequency_bins / args.bandwidth_hz

    for result in results:
        record_by_path = {rec.path_index: rec for rec in result.records}
        for rec in result.records:
            check1.append(rec.check1_residual_m)
            if not np.isnan(rec.check2_residual_m):
                check2.append(rec.check2_residual_m)
            ray_distances.append(rec.ray_distance_m)
            if not np.isnan(rec.check3_unfold_m):
                check3.append(rec.check3_unfold_m)
            if rec.is_image_source:
                image_source_paths += 1
                bounce_histogram[rec.bounce] = bounce_histogram.get(rec.bounce, 0) + 1
                if rec.has_refraction:
                    refraction_image_source_paths += 1
            else:
                non_image_source_paths += 1
        front_vs_count += sum(1 for rec in result.records if rec.in_front and rec.is_image_source)

        view_peak = max(float(np.max(result.center_power)), 1e-30)
        for peak, path_id, distance in zip(
            result.peaks, result.peak_match_path, result.peak_match_distance
        ):
            peak_count += 1
            peak_distances.append(float(distance))
            peak_power = float(result.center_power[peak])
            if peak_power >= view_peak * 10.0 ** (-10.0 / 10.0):
                strong_peak_distances.append(float(distance))
            if path_id < 0 or distance > beamwidth:
                unresolved += 1
                continue
            matched_peaks += 1
            row, col = peak
            profile = result.power_volume[row, col, :]
            dominant = float(result.delay_s[int(np.argmax(profile))])
            tau_ref = record_by_path[int(path_id)].tau_s % period
            delay_errors.append(circular_delay_error_s(dominant, tau_ref, period))

        strong_count += result.strong_source_count
        recalled_count += result.recalled_count
        resolvable_count += result.resolvable_count
        resolvable_recalled += result.resolvable_recalled
        for rec, recalled in zip(result.strong_sources, result.strong_recalled):
            if not bool(recalled):
                missed_strong.append(
                    {
                        "ue": rec.ue,
                        "path_index": rec.path_index,
                        "power_db": round(rec.power_db, 3),
                        "image_power_db": round(rec.image_power_db, 3),
                        "ky": round(rec.ky, 6),
                        "kz": round(rec.kz, 6),
                    }
                )
        per_view.append(
            {
                "name": result.name,
                "peak_count": len(result.peaks),
                "matched_peaks": result.matched_peaks,
                "precision": result.precision,
                "strong_source_count": result.strong_source_count,
                "recalled_count": result.recalled_count,
                "recall": result.strong_recall,
                "resolvable_count": result.resolvable_count,
                "resolvable_recalled": result.resolvable_recalled,
                "resolvable_recall": result.resolvable_recall,
            }
        )

    return {
        "num_paths": sum(len(r.records) for r in results),
        "image_source_paths": image_source_paths,
        "refraction_image_source_paths": refraction_image_source_paths,
        "non_image_source_paths": non_image_source_paths,
        "bounce_histogram": dict(sorted(bounce_histogram.items())),
        "check1_max_abs_m": float(np.max(np.abs(check1))) if check1 else float("nan"),
        "check2_max_abs_m": float(np.max(np.abs(check2))) if check2 else float("nan"),
        "check2_count": len(check2),
        "ray_distance_max_abs_m": (
            float(np.max(np.abs(ray_distances))) if ray_distances else float("nan")
        ),
        "check3_max_abs_m": float(np.max(np.abs(check3))) if check3 else float("nan"),
        "check3_count": len(check3),
        "peak_count": peak_count,
        "matched_peaks": matched_peaks,
        "precision": matched_peaks / peak_count if peak_count else float("nan"),
        "strong_peak_count": len(strong_peak_distances),
        "beamwidth": beamwidth,
        "resolution": resolution,
        "peak_match_median": float(np.median(peak_distances)) if peak_distances else float("nan"),
        "peak_match_max": float(np.max(peak_distances)) if peak_distances else float("nan"),
        "strong_peak_match_median": (
            float(np.median(strong_peak_distances)) if strong_peak_distances else float("nan")
        ),
        "strong_peak_match_max": (
            float(np.max(strong_peak_distances)) if strong_peak_distances else float("nan")
        ),
        "period_s": period,
        "delay_error_median_s": float(np.median(delay_errors)) if delay_errors else float("nan"),
        "delay_error_max_s": float(np.max(delay_errors)) if delay_errors else float("nan"),
        "delay_error_count": len(delay_errors),
        "unresolved_peaks": unresolved,
        "direct_vs_count": bounce_histogram.get(0, 0),
        "front_vs_count": front_vs_count,
        "strong_source_count": strong_count,
        "recalled_count": recalled_count,
        "recall": recalled_count / strong_count if strong_count else float("nan"),
        "resolvable_count": resolvable_count,
        "resolvable_recalled": resolvable_recalled,
        "resolvable_recall": (
            resolvable_recalled / resolvable_count if resolvable_count else float("nan")
        ),
        "per_view": per_view,
        "missed_strong_sources": missed_strong,
    }


def _format_optional(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.4f}"


def _json_safe(value: object) -> object:
    """Replace non-finite floats with ``None`` so summary.json stays strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(
    output_path: Path,
    *,
    boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    results: list[ViewResult],
    summary: dict[str, object],
    args: argparse.Namespace,
) -> None:
    """Write the human-readable markdown report."""
    lines: list[str] = []
    lines.append("# RF-camera image-source analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Scene: `sionna.rt.scene.simple_street_canyon`")
    lines.append(
        f"- Carrier: {args.carrier_hz / 1e9:.3f} GHz, bandwidth {args.bandwidth_hz / 1e6:.1f} MHz"
    )
    lines.append(f"- BS: {BS_POSITION} (look_at {BS_LOOK_AT})")
    lines.append(
        f"- Aperture: {args.rx_rows}x{args.rx_cols} at {args.spacing} lambda, "
        f"FFT {args.fft_rows}x{args.fft_cols}, {args.num_frequency_bins} frequency bins"
    )
    lines.append(f"- Aperture taper: {args.taper}")
    lines.append(
        f"- Beamwidth (matching gate) {float(summary['beamwidth']):.4f}; "
        f"tapered first-null resolution {float(summary['resolution']):.4f} in direction cosine"
    )
    lines.append(
        f"- Trace: max_depth={args.max_depth}, synthetic_array=True, seed={args.seed}, "
        "LoS + specular + refraction (no diffraction)"
    )
    lines.append("")
    lines.append("### Building bounding boxes")
    lines.append("")
    lines.append("| object | x_min | x_max | y_min | y_max | z_max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, low, high in boxes:
        lines.append(
            f"| {label} | {low[0]:.2f} | {high[0]:.2f} | {low[1]:.2f} | "
            f"{high[1]:.2f} | {high[2]:.2f} |"
        )
    lines.append("")
    lines.append("## Per-UE paths")
    lines.append("")
    for result in results:
        lines.append(
            f"### {result.name} at {tuple(round(v, 3) for v in result.position)}, "
            f"orientation {tuple(round(v, 4) for v in result.orientation)}"
        )
        lines.append("")
        lines.append(
            "| path | chain | bounces | refract | image src | power dB | image power dB | "
            "tau ns | VS (x,y,z) | front | ky/k | kz/k | kx | check1 [m] | check2 [m] | "
            "ray dist [m] | check3 [m] |"
        )
        lines.append(
            "| ---: | --- | ---: | :---: | :---: | ---: | ---: | ---: | --- | :---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for rec in result.records:
            chain = "-".join(str(i) for i in rec.interaction_chain)
            vs = f"({rec.vs[0]:.2f}, {rec.vs[1]:.2f}, {rec.vs[2]:.2f})"
            check2 = "n/a" if np.isnan(rec.check2_residual_m) else f"{rec.check2_residual_m:.2e}"
            check3 = "n/a" if np.isnan(rec.check3_unfold_m) else f"{rec.check3_unfold_m:.2e}"
            image_power = "n/a" if np.isnan(rec.image_power_db) else f"{rec.image_power_db:.1f}"
            lines.append(
                f"| {rec.path_index} | {chain} | {rec.bounce} | {rec.has_refraction} | "
                f"{rec.is_image_source} | {rec.power_db:.1f} | {image_power} | "
                f"{rec.tau_s * 1e9:.2f} | {vs} | "
                f"{rec.in_front} | {rec.ky:+.3f} | {rec.kz:+.3f} | {rec.kx:+.3f} | "
                f"{rec.check1_residual_m:.2e} | {check2} | {rec.ray_distance_m:.2e} | {check3} |"
            )
        lines.append("")
        if result.peaks:
            lines.append("Peak -> nearest front-hemisphere image source (direction cosine):")
            lines.append("")
            lines.append("| peak row | peak col | ky/k | kz/k | nearest path | distance |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
            for (row, col), path_id, distance in zip(
                result.peaks, result.peak_match_path, result.peak_match_distance
            ):
                nearest = "unmatched" if path_id < 0 else str(int(path_id))
                distance_text = "inf" if np.isinf(distance) else f"{distance:.4f}"
                lines.append(
                    f"| {row} | {col} | {result.ky_over_k[col]:+.3f} "
                    f"| {result.kz_over_k[row]:+.3f} | {nearest} | {distance_text} |"
                )
            lines.append("")
        # Delay at each matched peak versus its matched image source.
        record_by_path = {rec.path_index: rec for rec in result.records}
        period = float(summary["period_s"])
        matched = [
            (peak, int(path_id))
            for peak, path_id in zip(result.peaks, result.peak_match_path)
            if path_id >= 0
        ]
        if matched:
            lines.append("Dominant delay at each matched peak versus image-source delay:")
            lines.append("")
            lines.append("| peak | dominant delay ns | VS tau mod period ns | circular error ns |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for peak, path_id in matched:
                row, col = peak
                dominant = float(result.delay_s[int(np.argmax(result.power_volume[row, col, :]))])
                tau_ref = record_by_path[path_id].tau_s % period
                error = circular_delay_error_s(dominant, tau_ref, period)
                lines.append(
                    f"| {peak} | {dominant * 1e9:.1f} | {tau_ref * 1e9:.1f} | {error * 1e9:.1f} |"
                )
            lines.append("")
        lines.append("Recall of strong front-hemisphere image sources:")
        lines.append("")
        lines.append(
            f"- Strong sources: {result.strong_source_count}; recalled: "
            f"{result.recalled_count} ({_format_optional(result.strong_recall)}); "
            f"precision {_format_optional(result.precision)}"
        )
        lines.append(
            f"- Resolvable strong sources: {result.resolvable_count}; recalled: "
            f"{result.resolvable_recalled} ({_format_optional(result.resolvable_recall)})"
        )
        missed = [
            rec
            for rec, recalled in zip(result.strong_sources, result.strong_recalled)
            if not bool(recalled)
        ]
        if missed:
            lines.append("- Missed strong sources:")
            lines.append("")
            lines.append("| path | power dB | image power dB | ky/k | kz/k |")
            lines.append("| ---: | ---: | ---: | ---: | ---: |")
            for rec in missed:
                lines.append(
                    f"| {rec.path_index} | {rec.power_db:.1f} | {rec.image_power_db:.1f} | "
                    f"{rec.ky:+.3f} | {rec.kz:+.3f} |"
                )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Valid paths: {summary['num_paths']} ({summary['bounce_histogram']} by bounce order)"
    )
    lines.append(f"- Image-source paths: {summary['image_source_paths']}")
    lines.append(
        f"- Refraction-containing image-source paths: {summary['refraction_image_source_paths']}"
    )
    lines.append(
        f"- Non-image-source paths (diffuse/diffraction): {summary['non_image_source_paths']}"
    )
    lines.append(f"- Front-hemisphere image-source paths: {summary['front_vs_count']}")
    lines.append(
        f"- Check 1 (polyline length vs c*tau) max abs residual: "
        f"{float(summary['check1_max_abs_m']):.3e} m"
    )
    lines.append(
        f"- Check 2 (single bounce |v1-BS| == |v1-VS|) max abs residual over "
        f"{summary['check2_count']} paths: {float(summary['check2_max_abs_m']):.3e} m"
    )
    lines.append(
        f"- Ray distance (last vertex to UE arrival ray) max abs residual: "
        f"{float(summary['ray_distance_max_abs_m']):.3e} m"
    )
    lines.append(
        f"- Check 3 (vertex unfolding vs tau/angle VS) max abs residual over "
        f"{summary['check3_count']} image-source paths: "
        f"{float(summary['check3_max_abs_m']):.3e} m"
    )
    lines.append(
        f"- Peaks found: {summary['peak_count']}; matched: {summary['matched_peaks']} "
        f"(precision {float(summary['precision']):.4f}); unresolved: "
        f"{summary['unresolved_peaks']}"
    )
    lines.append(
        f"- Peak/VS direction-cosine distance, all peaks: median "
        f"{float(summary['peak_match_median']):.4f}, max {float(summary['peak_match_max']):.4f}"
    )
    lines.append(
        f"- Peak/VS direction-cosine distance, main-lobe peaks (>= -10 dB): median "
        f"{float(summary['strong_peak_match_median']):.4f}, "
        f"max {float(summary['strong_peak_match_max']):.4f} over "
        f"{summary['strong_peak_count']} peaks"
    )
    period_ns = float(summary["period_s"]) * 1e9
    lines.append(
        f"- Delay error (dominant delay vs matched VS tau, modulo {period_ns:.1f} ns): "
        f"median {float(summary['delay_error_median_s']) * 1e9:.1f} ns, "
        f"max {float(summary['delay_error_max_s']) * 1e9:.1f} ns over "
        f"{summary['delay_error_count']} matched peaks"
    )
    lines.append(
        f"- Strong-source recall: {summary['recalled_count']} / "
        f"{summary['strong_source_count']} ({float(summary['recall']):.4f})"
    )
    lines.append(
        f"- Resolvable strong-source recall: {summary['resolvable_recalled']} / "
        f"{summary['resolvable_count']} ({float(summary['resolvable_recall']):.4f})"
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Matching gate (beamwidth) is 1/(N*d) = "
        f"{float(summary['beamwidth']):.3f} in direction cosine; with taper "
        f"'{args.taper}' the first-null (Rayleigh) resolution is "
        f"{float(summary['resolution']):.3f}, so the resolvable subset uses that separation."
    )
    lines.append(
        f"- One delay bin is {1.0 / args.bandwidth_hz * 1e9:.1f} ns; delays repeat modulo "
        f"{float(summary['period_s']) * 1e9:.1f} ns."
    )
    lines.append(
        "- The image is A = kx*U, so a source's image power is its path power times kx^2; "
        "strong sources are selected on this kx^2-weighted power so that the recall threshold "
        "matches the peak threshold."
    )
    lines.append(
        "- Back-hemisphere arrivals are excluded from the developed image and from the "
        "candidate set; only their geometry is listed."
    )
    lines.append(
        "- Sionna models transmission as straight-through (thin slab, no bending), so "
        "LoS/specular/refraction chains all form image sources; only diffuse and "
        "diffraction paths do not."
    )
    lines.append(
        "- Virtual-source positions come from paths.tau and world-frame paths.theta_r/phi_r "
        "and rely on Sionna's arrival-angle convention (theta_r/phi_r point from the UE toward "
        "the source of the wave)."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scene = load_scene(simple_street_canyon)
    scene.frequency = args.carrier_hz
    print("=== Scene object bounding boxes ===")
    boxes = building_boxes(scene)

    configure_rf_camera_arrays(
        scene,
        rx_rows=args.rx_rows,
        rx_cols=args.rx_cols,
        vertical_spacing_lambda=args.spacing,
        horizontal_spacing_lambda=args.spacing,
        tx_pattern="iso",
        rx_pattern=HEMISPHERE_SPLIT_PATTERN,
    )
    scene.add(Transmitter(name="bs", position=list(BS_POSITION), look_at=list(BS_LOOK_AT)))
    orientations = {}
    for name, position, look_at in UE_VIEWS:
        orientation = look_at_orientation(position, look_at)
        orientations[name] = orientation
        scene.add(Receiver(name=name, position=list(position), orientation=list(orientation)))

    print("=== Path tracing ===")
    paths = trace_paths(scene, max_depth=args.max_depth, synthetic_array=True, seed=args.seed)
    frequency_offsets_hz = frequency_offsets(args.bandwidth_hz, args.num_frequency_bins)
    apertures = aperture_cfrs(
        paths,
        frequency_offsets_hz,
        num_rx=len(UE_VIEWS),
        rx_rows=args.rx_rows,
        rx_cols=args.rx_cols,
    )

    tau = np.asarray(paths.tau)
    theta_r = np.asarray(paths.theta_r)
    phi_r = np.asarray(paths.phi_r)
    valid = np.asarray(paths.valid)
    vertices = np.asarray(paths.vertices)
    interactions = np.asarray(paths.interactions)
    a_real, a_imag = paths.a
    a_real = np.asarray(a_real)
    a_imag = np.asarray(a_imag)
    print(
        "path shapes: "
        f"tau={tau.shape}, vertices={vertices.shape}, interactions={interactions.shape}, "
        f"a={a_real.shape}"
    )
    if tau.shape[:2] != (len(UE_VIEWS), 1):
        raise RuntimeError(f"Expected tau with shape [num_rx, 1, num_paths], got {tau.shape}")
    if vertices.shape[1:3] != (len(UE_VIEWS), 1):
        raise RuntimeError(f"Unexpected vertices shape {vertices.shape}")
    if a_real.shape[:2] != (len(UE_VIEWS), 2 * args.rx_rows * args.rx_cols):
        raise RuntimeError(f"Unexpected a shape {a_real.shape}")

    # Isotropic-element path power: front element 0 plus back element 0 (the two
    # hemisphere patterns have disjoint support, so their sum is the full element).
    a = a_real + 1j * a_imag
    path_power = pattern_summed_element_power(
        a[:, :, 0, 0, :], rows=args.rx_rows, cols=args.rx_cols, axis=1
    )

    results: list[ViewResult] = []
    for rx_index, (name, position, _look_at) in enumerate(UE_VIEWS):
        result = analyze_view(
            name=name,
            position=position,
            orientation=orientations[name],
            rx_index=rx_index,
            aperture=apertures[rx_index],
            tau=tau,
            theta_r=theta_r,
            phi_r=phi_r,
            valid=valid,
            vertices=vertices,
            interactions=interactions,
            path_power=path_power,
            frequency_offsets_hz=frequency_offsets_hz,
            args=args,
        )
        results.append(result)
        print(f"  {name}: {len(result.records)} valid paths, {len(result.peaks)} image peaks")

    plot_topview(args.out / "topview.png", boxes, results)

    peak = max(float(np.max(np.abs(r.center_power[r.valid_mask]))) for r in results)
    for result in results:
        records = [rec for rec in result.records if rec.is_image_source]
        markers = []
        labelled: set[int] = set()
        for rec in records:
            if not rec.in_front:
                continue
            # Label each bounce order once; leading "_" keeps duplicates out of the legend.
            label = f"b{rec.bounce}" if rec.bounce not in labelled else f"_b{rec.bounce}"
            labelled.add(rec.bounce)
            markers.append(Marker(rec.ky, rec.kz, "x", label))
        image = np.ma.masked_where(
            ~result.valid_mask,
            normalized_power_db(
                result.center_power, max(float(np.max(result.center_power)), 1e-30)
            ),
        )
        save_direction_image(
            image,
            args.out / f"{result.name}_image.png",
            extent=image_extent(result.ky_over_k, result.kz_over_k),
            title=(
                f"{result.name}: front |A|^2 at center frequency, "
                "markers = front-hemisphere image sources"
            ),
            colorbar_label="dB relative to view peak",
            vmin=-60.0,
            vmax=0.0,
            markers=markers,
            figsize=(7, 6),
            dpi=120,
        )
        print(
            f"  {result.name}: peak view power {np.max(result.center_power):.3e}, global {peak:.3e}"
        )

    write_csv(args.out / "virtual_sources.csv", results)
    summary = build_summary(results, args)
    write_report(args.out / "report.md", boxes=boxes, results=results, summary=summary, args=args)

    summary_json = json.dumps(_json_safe(summary), indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(summary_json, encoding="utf-8")
    print("=== Summary ===")
    print(summary_json)
    print(f"outputs written to {args.out}")


if __name__ == "__main__":
    main()
