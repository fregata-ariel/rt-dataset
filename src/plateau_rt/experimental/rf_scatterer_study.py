"""CLI study: identifiability of RF point-scatterer reconstruction.

Synthesises the aperture CFR of a fixed set of isotropic point scatterers on a
regular voxel grid, reconstructs their complex reflectivities by regularised
least squares, and sweeps the number of ring views, the aperture size and the
bandwidth. For every configuration it reports the reconstruction error of the
noisy, bias-only, oracle and noiseless estimates, the conditioning of the system
matrix and how often the true scatterer voxels land in the strongest
reflectivity bins.

Run as::

    python -m plateau_rt.experimental.rf_scatterer_study --out DIR [--seed 0] [--quick]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from plateau_rt.domain.rf_camera.camera import RFViewSpec, generate_ring_views
from plateau_rt.experimental.rf_scatterer_fit import (
    ApertureSpec,
    direct_path_cfr,
    discrepancy_damp,
    reconstruct,
    relative_error,
    scatterer_cfr,
    system_matrix,
    tikhonov_path,
    voxel_axes,
    voxel_grid,
)

CARRIER_HZ = 3.5e9
NUM_FREQ = 16
SNR_DB = 30.0
MAX_DENSE_ENTRIES = 200_000_000
NUM_DAMPS = 33
DAMP_MIN_REL = 1e-8
DAMP_MAX_REL = 1e0
NOISELESS_DAMP_REL = 1e-10

BS_POSITION = (-50.0, -50.0, 30.0)
TARGET = (0.0, 0.0, 5.0)
RADIUS_M = 30.0
UE_HEIGHT_M = 1.5
VOXEL_SPACING_M = 2.0
VOXEL_BOUNDS_MIN = (-5.0, -5.0, 0.0)
VOXEL_BOUNDS_MAX = (5.0, 5.0, 10.0)

FULL_VIEWS = (1, 2, 4, 8, 16)
FULL_APERTURES = ((4, 4), (8, 8), (16, 16))
FULL_BANDWIDTHS_HZ = (20e6, 100e6, 400e6)

QUICK_VIEWS = (1, 4)
QUICK_APERTURES = ((4, 4), (8, 8))
QUICK_BANDWIDTHS_HZ = (100e6,)

PLOT_BANDWIDTH_HZ = 100e6

# Ground-truth scatterers sit on the faces of the x,y in [-5, 5], z in [0, 10]
# box, exactly on voxel centres. |rho|**2 is the bistatic RCS in m^2.
SCATTERER_VOXEL_INDICES = ((5, 3, 2), (0, 1, 4), (3, 5, 1), (1, 0, 5))
SCATTERER_RHO = np.array(
    [1.0 + 0.0j, 0.8 * np.exp(0.7j), 1.2 * np.exp(-1.1j), 0.6 * np.exp(2.0j)],
    dtype=np.complex128,
)


@dataclass(frozen=True)
class SweepConfig:
    """One point of the (views, aperture, bandwidth) sweep."""

    num_views: int
    rows: int
    cols: int
    bandwidth_hz: float

    @property
    def aperture_label(self) -> str:
        return f"{self.rows}x{self.cols}"

    @property
    def bandwidth_label(self) -> float:
        return self.bandwidth_hz / 1e6


def build_scene() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Return ``(points, rho_true, axes, voxels)`` for the fixed ground truth."""
    axes = voxel_axes(VOXEL_BOUNDS_MIN, VOXEL_BOUNDS_MAX, VOXEL_SPACING_M)
    voxels = voxel_grid(VOXEL_BOUNDS_MIN, VOXEL_BOUNDS_MAX, VOXEL_SPACING_M)
    points = np.array(
        [[axes[0][i], axes[1][j], axes[2][k]] for (i, j, k) in SCATTERER_VOXEL_INDICES],
        dtype=np.float64,
    )
    rho_true = np.zeros(voxels.shape[0], dtype=np.complex128)
    for point, rho in zip(points, SCATTERER_RHO):
        index = int(np.argmin(np.linalg.norm(voxels - point, axis=1)))
        if not np.allclose(voxels[index], point):
            raise RuntimeError("scatterer is not on a voxel centre")
        rho_true[index] += rho
    return points, rho_true, axes, voxels


def _views(num_views: int) -> list[RFViewSpec]:
    return generate_ring_views(
        target=TARGET,
        radius_m=RADIUS_M,
        ue_height_m=UE_HEIGHT_M,
        num_views=num_views,
    )


def _top_hit_fraction(rho_hat: np.ndarray, rho_true: np.ndarray) -> float:
    true_indices = set(np.flatnonzero(np.abs(rho_true) > 0.0).tolist())
    num_true = len(true_indices)
    if num_true == 0:
        raise ValueError("ground truth has no scatterers")
    top = np.argsort(-np.abs(rho_hat))[:num_true]
    hits = len(true_indices & set(top.tolist()))
    return hits / num_true


def _evaluate(
    config: SweepConfig,
    views: list[RFViewSpec],
    rho_true: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Evaluate one configuration; returns its record and the noisy estimate.

    The noisy estimate uses the damp chosen by the Morozov discrepancy
    principle with the true noise sigma. ``bias_error`` reuses that same damp on
    the noiseless scatterer data, ``oracle_error`` is the best noisy-data error
    over the damp grid, and ``noiseless_error`` uses a tiny full-rank damp.
    """
    spec = ApertureSpec(
        rows=config.rows,
        cols=config.cols,
        carrier_hz=CARRIER_HZ,
        bandwidth_hz=config.bandwidth_hz,
        num_freq=NUM_FREQ,
    )
    base = {
        "views": config.num_views,
        "aperture": config.aperture_label,
        "bandwidth_mhz": config.bandwidth_label,
    }
    voxels = voxel_grid(VOXEL_BOUNDS_MIN, VOXEL_BOUNDS_MAX, VOXEL_SPACING_M)
    matrix = system_matrix(spec, BS_POSITION, views, voxels)

    if matrix.size > MAX_DENSE_ENTRIES:
        return {**base, "status": "skipped"}, None

    true_indices = np.flatnonzero(np.abs(rho_true) > 0.0)
    points = voxels[true_indices]
    rho = rho_true[true_indices]
    y_scatter = np.concatenate(
        [
            scatterer_cfr(spec, BS_POSITION, view.position, view.orientation, points, rho).reshape(
                -1
            )
            for view in views
        ]
    )
    y_direct = np.concatenate(
        [
            direct_path_cfr(spec, BS_POSITION, view.position, view.orientation).reshape(-1)
            for view in views
        ]
    )

    mean_power = float(np.mean(np.abs(y_scatter) ** 2))
    sigma = float(np.sqrt(mean_power / (10.0 ** (SNR_DB / 10.0))))
    noise = (
        sigma
        / np.sqrt(2.0)
        * (rng.standard_normal(y_scatter.shape) + 1j * rng.standard_normal(y_scatter.shape))
    )

    svd = np.linalg.svd(matrix, full_matrices=False)
    s_max = float(svd[1][0])
    s_min = float(svd[1][-1])
    damps = s_max * np.logspace(np.log10(DAMP_MIN_REL), np.log10(DAMP_MAX_REL), NUM_DAMPS)

    # Background subtraction: the known direct path is removed from the full
    # observation before inversion (exact in float64 to ~1e-19).
    y_noisy = (y_direct + y_scatter + noise) - y_direct
    noisy_path = tikhonov_path(matrix, y_noisy, damps, svd=svd)
    scatter_path = tikhonov_path(matrix, y_scatter, damps, svd=svd)
    chosen = discrepancy_damp(matrix, y_noisy, sigma, damps, svd=svd)
    chosen_index = int(np.argmin(np.abs(damps - chosen)))

    noisy_estimate = noisy_path[chosen_index]
    noisy_error = relative_error(noisy_estimate, rho_true)
    bias_error = relative_error(scatter_path[chosen_index], rho_true)
    oracle_errors = [relative_error(noisy_path[index], rho_true) for index in range(damps.size)]
    oracle_index = int(np.argmin(oracle_errors))
    noiseless = reconstruct(matrix, y_scatter, damp=NOISELESS_DAMP_REL * s_max, method="normal")

    record = {
        **base,
        "cond": s_max / s_min,
        "damp_rel": chosen / s_max,
        "noisy_error": noisy_error,
        "bias_error": bias_error,
        "oracle_error": float(oracle_errors[oracle_index]),
        "oracle_damp_rel": float(damps[oracle_index] / s_max),
        "noiseless_error": relative_error(noiseless, rho_true),
        "hit_fraction": _top_hit_fraction(noisy_estimate, rho_true),
        "direct_over_scatter_db": 10.0
        * float(np.log10(np.mean(np.abs(y_direct) ** 2) / mean_power)),
        "status": "ok",
    }
    return record, noisy_estimate


def evaluate_config(config: SweepConfig, seed: int) -> dict[str, Any]:
    """Evaluate one configuration from a fresh seed; returns its metric record."""
    _, rho_true, _, _ = build_scene()
    rng = np.random.default_rng(seed)
    record, _ = _evaluate(config, _views(config.num_views), rho_true, rng)
    return record


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rho_grid(estimate: np.ndarray, axes: tuple[np.ndarray, ...]) -> np.ndarray:
    shape = (axes[0].size, axes[1].size, axes[2].size)
    return np.abs(np.asarray(estimate).reshape(shape))


def _save_reconstruction_maps(
    output_dir: Path,
    title: str,
    estimate: np.ndarray,
    truth: np.ndarray,
    axes: tuple[np.ndarray, ...],
    filename: str,
) -> Path:
    plt = _pyplot()
    estimate_grid = _rho_grid(estimate, axes)
    truth_grid = _rho_grid(truth, axes)

    panels = (
        ("truth: |rho| x-y", truth_grid.max(axis=2), axes[0], axes[1]),
        ("truth: |rho| x-z", truth_grid.max(axis=1), axes[0], axes[2]),
        ("estimate: |rho| x-y", estimate_grid.max(axis=2), axes[0], axes[1]),
        ("estimate: |rho| x-z", estimate_grid.max(axis=1), axes[0], axes[2]),
    )
    figure, axes_plot = plt.subplots(2, 2, figsize=(9.0, 8.0))
    for axis_plot, (panel_title, image, horizontal, vertical) in zip(axes_plot.ravel(), panels):
        im = axis_plot.imshow(
            image.T,
            origin="lower",
            extent=[horizontal[0], horizontal[-1], vertical[0], vertical[-1]],
            aspect="auto",
        )
        axis_plot.set_title(panel_title)
        axis_plot.set_xlabel("world axis 1 [m]")
        axis_plot.set_ylabel("world axis 2 [m]")
        figure.colorbar(im, ax=axis_plot, label="|rho|")
    figure.suptitle(title)
    figure.tight_layout()
    path = output_dir / filename
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _save_error_vs_views(output_dir: Path, records: list[dict[str, Any]]) -> Path | None:
    selected = [record for record in records if record.get("status") == "ok"]
    if not selected:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    apertures = sorted({record["aperture"] for record in selected})
    for aperture in apertures:
        rows = sorted(
            (
                record
                for record in selected
                if record["aperture"] == aperture
                and record["bandwidth_mhz"] == PLOT_BANDWIDTH_HZ / 1e6
            ),
            key=lambda record: record["views"],
        )
        if not rows:
            continue
        views = [record["views"] for record in rows]
        axis.plot(
            views,
            [record["noisy_error"] for record in rows],
            marker="o",
            linestyle="-",
            label=f"{aperture} noisy",
        )
        axis.plot(
            views,
            [record["bias_error"] for record in rows],
            marker="s",
            linestyle="--",
            label=f"{aperture} bias (same damp)",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(sorted({record["views"] for record in selected}))
    axis.set_xticklabels([str(value) for value in sorted({record["views"] for record in selected})])
    axis.set_xlabel("number of views")
    axis.set_ylabel("relative error (30 dB)")
    axis.set_title(f"Error vs views at {PLOT_BANDWIDTH_HZ / 1e6:.0f} MHz")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="aperture")
    figure.tight_layout()
    path = output_dir / "error_vs_views.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _format_optional(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}e}"


def _format_db(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.1f}"


def _markdown_table(records: list[dict[str, Any]]) -> str:
    header = (
        "| views | aperture | bandwidth [MHz] | cond | damp_rel | noisy error | bias error "
        "| oracle error (uses truth) | oracle damp_rel | noiseless error | hit fraction "
        "| direct/scatter [dB] | status |"
    )
    separator = "|" + "---|" * 13
    lines = [header, separator]
    for record in records:
        hit = record.get("hit_fraction")
        hit_text = f"{hit:.2f}" if hit is not None else "-"
        cells = [
            str(record["views"]),
            str(record["aperture"]),
            f"{record['bandwidth_mhz']:.0f}",
            _format_optional(record.get("cond")),
            _format_optional(record.get("damp_rel")),
            _format_optional(record.get("noisy_error")),
            _format_optional(record.get("bias_error")),
            _format_optional(record.get("oracle_error")),
            _format_optional(record.get("oracle_damp_rel")),
            _format_optional(record.get("noiseless_error")),
            hit_text,
            _format_db(record.get("direct_over_scatter_db")),
            str(record["status"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    artifacts: dict[str, tuple[SweepConfig, np.ndarray]],
    rho_true: np.ndarray,
    axes: tuple[np.ndarray, ...],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def key_of(record: dict[str, Any]) -> str:
        return f"{record['views']}_{record['aperture']}_{float(record['bandwidth_mhz']):.0f}"

    ok_records = [record for record in records if record.get("status") == "ok"]
    png_paths: list[Path] = []
    if ok_records:
        best_key = min(ok_records, key=lambda record: record["noisy_error"])
        worst_key = max(ok_records, key=lambda record: record["noisy_error"])
        for label, record in (("best", best_key), ("worst", worst_key)):
            artifact = artifacts.get(key_of(record))
            if artifact is None:
                continue
            config, estimate = artifact
            png_paths.append(
                _save_reconstruction_maps(
                    output_dir,
                    (
                        f"{label} configuration: {config.num_views} views, "
                        f"{config.aperture_label} aperture, {config.bandwidth_label:.0f} MHz "
                        f"(noisy relative error {record['noisy_error']:.3e})"
                    ),
                    estimate,
                    rho_true,
                    axes,
                    f"reconstruction_{label}.png",
                )
            )

    error_png = _save_error_vs_views(output_dir, records)

    note = (
        "Each configuration reports four errors. `noisy_error` uses the damp "
        "chosen by the Morozov discrepancy principle on the noisy data; "
        "`bias_error` reuses that same damp on the noiseless scatterer data, so "
        "it isolates the regularisation bias from the noise amplification; "
        "`oracle_error` is the best noisy-data error over the damp grid and "
        "explicitly uses the truth, so it is a lower bound rather than a "
        "deployable estimate. `noiseless_error` uses a tiny full-rank damp. "
        "The study has firm limits: the ground-truth scatterers sit exactly on "
        "voxel centres and the same forward model generates and inverts the "
        "data, so there is no gridding or model mismatch (the off-grid unit "
        "test covers that separately); background subtraction is exact in "
        "float64 (the subtraction residual is ~2e-19), so the direct path costs "
        "nothing; the SNR is relative to the scatterer power alone while the "
        "direct path is tens of dB stronger (about 36 dB at 8v/8x8/100 MHz, see "
        "`direct_over_scatter_db`); and there is no multipath, occlusion, "
        "antenna pattern or front/back hemisphere split. The discrepancy "
        "principle with tau = 1 is conservative: in several mid-conditioned "
        "configurations (e.g. 8 views, 8x8, 100 MHz) `bias_error` is still close "
        "to `noisy_error` and the oracle is noticeably lower, so those noisy "
        "errors partly reflect the damp choice rather than the noise floor."
    )
    report = [
        "# RF scatterer reconstruction study",
        "",
        f"Scene: {len(SCATTERER_RHO)} isotropic point scatterers on voxel centres "
        f"(spacing {VOXEL_SPACING_M:.1f} m), BS at {BS_POSITION}, "
        f"ring views around {TARGET} at {RADIUS_M:.0f} m.",
        f"Noise: complex Gaussian at {SNR_DB:.0f} dB relative to the mean scatterer power.",
        "",
        _markdown_table(records),
        "",
        "## Note",
        "",
        note,
        "",
        "## Figures",
        "",
        *[f"- `{path.name}`" for path in [*png_paths, error_png] if path is not None],
        "",
    ]
    (output_dir / "results.md").write_text("\n".join(report), encoding="utf-8")

    payload = {
        "seed": seed,
        "snr_db": SNR_DB,
        "carrier_hz": CARRIER_HZ,
        "num_freq": NUM_FREQ,
        "voxel_spacing_m": VOXEL_SPACING_M,
        "voxel_bounds_min": list(VOXEL_BOUNDS_MIN),
        "voxel_bounds_max": list(VOXEL_BOUNDS_MAX),
        "bs_position": list(BS_POSITION),
        "target": list(TARGET),
        "radius_m": RADIUS_M,
        "ue_height_m": UE_HEIGHT_M,
        "scatterers": [
            {"position_m": [float(v) for v in point], "rho": [rho.real, rho.imag]}
            for point, rho in zip(
                np.array(
                    [[axes[0][i], axes[1][j], axes[2][k]] for (i, j, k) in SCATTERER_VOXEL_INDICES]
                ),
                SCATTERER_RHO,
            )
        ],
        "records": records,
        "figures": [path.name for path in [*png_paths, error_png] if path is not None],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_study(output_dir: Path, seed: int, quick: bool) -> list[dict[str, Any]]:
    """Run the sweep and write results; returns the metric records."""
    if quick:
        views_sweep, apertures, bandwidths = QUICK_VIEWS, QUICK_APERTURES, QUICK_BANDWIDTHS_HZ
    else:
        views_sweep, apertures, bandwidths = FULL_VIEWS, FULL_APERTURES, FULL_BANDWIDTHS_HZ

    _, rho_true, axes, _ = build_scene()
    rng = np.random.default_rng(seed)
    views_cache = {num: _views(num) for num in views_sweep}

    records: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[SweepConfig, np.ndarray]] = {}
    for num_views in views_sweep:
        for rows, cols in apertures:
            for bandwidth in bandwidths:
                config = SweepConfig(num_views, rows, cols, bandwidth)
                record, estimate = _evaluate(config, views_cache[num_views], rho_true, rng)
                records.append(record)
                if estimate is not None:
                    artifacts[f"{num_views}_{rows}x{cols}_{bandwidth / 1e6:.0f}"] = (
                        config,
                        estimate,
                    )
                click.echo(
                    f"views={num_views:>2} aperture={rows}x{cols} "
                    f"bandwidth={bandwidth / 1e6:.0f} MHz"
                )

    _write_outputs(output_dir, records, artifacts, rho_true, axes, seed)
    return records


@click.command()
@click.option(
    "--out",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for results.md, results.json and the PNG maps.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Noise RNG seed.")
@click.option("--quick", is_flag=True, help="Run a tiny subset in under 30 s.")
def main(output_dir: Path, seed: int, quick: bool) -> None:
    """Run the RF point-scatterer identifiability study."""
    run_study(output_dir, seed, quick)
    click.echo(f"wrote study outputs to {output_dir}")


if __name__ == "__main__":
    main()
