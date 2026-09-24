"""Compare the analytic direct-path CFR with a Sionna RF-camera mock dataset.

Reads a dataset manifest (``dataset_manifest.json``) through
:mod:`plateau_rt.application.rf_dataset_manifest`, which supports both schema
v2 (single BS) and schema v3 (several base stations). One comparison is
reported per (view, BS) pair: the measured signal is that BS's aperture slice
summed over the front and back hemispheres to recover the isotropic element,
and it is correlated with
:func:`plateau_rt.experimental.rf_scatterer_fit.direct_path_cfr` using the
plane-wave model (Sionna ``synthetic_array=True``).

Sionna tracing is not bit-reproducible, so the script reports numbers instead
of asserting on them. The measured/model amplitude ratio is predicted from the
Tx 3GPP TR 38.901 field amplitude in the UE direction times the V/V
polarisation factor, both computed in NumPy here. A pair whose
``ratio_over_predicted`` departs from 1 by more than 1 dB is flagged as an
outlier.

In the mock (box building ``[0, 10]^3``, ring target ``(5, 5, 5)``), pairs
whose straight BS -> UE segment crosses the box are not true direct paths:
because the tracer enables refraction and Sionna traces refracted rays
without deflection, such a pair keeps the exact LoS delay (correlation ~1,
since correlation ignores magnitude) but has a large penetration loss, and it
is flagged as an outlier. Pass ``--blocker-aabb 0 0 0 10 10 10`` to also flag
geometrically blocked segments.

Pairs can also pick up specular reflections off the box: those keep a
near-predicted amplitude but lower the normalised correlation, so a pair is
flagged when its correlation is below ``--min-correlation`` (default 0.999).
``num_valid_paths`` (from ``path_geometry_gt.npz``, when readable) shows how
many traced paths the pair has. Normalised correlation is insensitive to
magnitude, so a refracted-only pair still correlates at ~1 and is caught by
the ratio test instead.

Run as::

    python -m plateau_rt.experimental.compare_direct_path [--dataset DIR]
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import click
import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    RFDatasetManifest,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.experimental.rf_scatterer_fit import (
    ApertureSpec,
    direct_path_cfr,
    frequencies_hz,
)

DEFAULT_DATASET = Path("data/generated/mock_results/rf_camera_multiview")

_THETA_3DB_DEG = 65.0
_PHI_3DB_DEG = 65.0
_SLA_V_DB = 30.0
_A_MAX_DB = 30.0
_G_E_MAX_DBI = 8.0


def _normalised_correlation(measured: np.ndarray, model: np.ndarray) -> float:
    a = np.asarray(measured, dtype=np.complex128).reshape(-1)
    b = np.asarray(model, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(abs(np.vdot(a, b)) / denominator)


def tr38901_amplitude(theta: float, phi: float) -> float:
    """Return the TR 38.901 element field amplitude for a direction.

    ``theta`` is the polar angle from local +z and ``phi`` the azimuth from
    local +x. The element uses ``theta_3dB = phi_3dB = 65 deg``,
    ``SLA_V = A_max = 30 dB`` and ``G_E,max = 8 dBi``; the returned value is
    ``sqrt(10**(A_dB / 10))``.
    """
    theta_3db = np.deg2rad(_THETA_3DB_DEG)
    phi_3db = np.deg2rad(_PHI_3DB_DEG)
    a_v = -min(12.0 * ((theta - 0.5 * np.pi) / theta_3db) ** 2, _SLA_V_DB)
    a_h = -min(12.0 * (phi / phi_3db) ** 2, _A_MAX_DB)
    a_db = -min(-(a_v + a_h), _A_MAX_DB) + _G_E_MAX_DBI
    return float(np.sqrt(10.0 ** (a_db / 10.0)))


def _theta_hat(theta: float, phi: float) -> np.ndarray:
    """Return the local ``theta`` unit vector of a spherical direction."""
    return np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        dtype=np.float64,
    )


def _spherical_angles(direction_local: np.ndarray) -> tuple[float, float]:
    theta = float(np.arccos(np.clip(direction_local[2], -1.0, 1.0)))
    phi = float(np.arctan2(direction_local[1], direction_local[0]))
    return theta, phi


def direct_path_ratio_prediction(
    bs: tuple[float, float, float],
    tx_look_at: tuple[float, float, float],
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return ``(tx_gain_amplitude, polarization_factor, predicted_ratio)``.

    The prediction assumes the Sionna ``tr38901`` transmit pattern and ``V``
    polarisation with an isotropic V receive element. The transmit gain is the
    field amplitude in the departure direction; the polarisation factor is
    ``|theta_hat_t_world . theta_hat_r_world|``.
    """
    bs_arr = np.asarray(bs, dtype=np.float64)
    ue_arr = np.asarray(ue_position, dtype=np.float64)

    tx_orientation = look_at_orientation(bs, tx_look_at)
    rotation_t = rotation_matrix(tx_orientation)
    rotation_r = rotation_matrix(ue_orientation)

    departure_world = ue_arr - bs_arr
    departure_world = departure_world / np.linalg.norm(departure_world)
    theta_t, phi_t = _spherical_angles(rotation_t.T @ departure_world)
    tx_gain = tr38901_amplitude(theta_t, phi_t)

    arrival_world = bs_arr - ue_arr
    arrival_world = arrival_world / np.linalg.norm(arrival_world)
    theta_r, phi_r = _spherical_angles(rotation_r.T @ arrival_world)

    theta_hat_t = rotation_t @ _theta_hat(theta_t, phi_t)
    theta_hat_r = rotation_r @ _theta_hat(theta_r, phi_r)
    polarization = abs(float(np.dot(theta_hat_t, theta_hat_r)))

    return tx_gain, polarization, tx_gain * polarization


def segment_intersects_aabb(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> bool:
    """Return whether the segment ``p0 -> p1`` intersects an axis-aligned box.

    Slab method over the parameter ``t in [0, 1]``; a segment that only touches
    the boundary counts as intersecting.
    """
    start = np.asarray(p0, dtype=np.float64)
    end = np.asarray(p1, dtype=np.float64)
    low = np.asarray(box_min, dtype=np.float64)
    high = np.asarray(box_max, dtype=np.float64)
    direction = end - start

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return False
            continue
        inverse = 1.0 / direction[axis]
        t1 = (low[axis] - start[axis]) * inverse
        t2 = (high[axis] - start[axis]) * inverse
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _load_valid_path_counts(manifest: RFDatasetManifest) -> np.ndarray | None:
    """Return the ``valid`` path-GT array, or None when unavailable or unexpected.

    ``valid`` must be stored as ``[view, bs, path]``: the canonical layout
    described by ``path_schema.json`` (or the reader's legacy fallback for
    synthetic-array datasets without a schema). This is a best-effort
    diagnostic: a missing or unreadable file or schema, another axis order
    (e.g. the ``--explicit-array`` native layout) or an unexpected shape
    yields None instead of raising.
    """
    try:
        gt = manifest.path_geometry_gt
        if gt is None:
            return None
        if gt.array_axes("valid") != ("view", "bs", "path"):
            return None
        if not gt.path.exists():
            return None
        with np.load(gt.path) as data:
            if "valid" not in data:
                return None
            valid = np.asarray(data["valid"])
        if valid.ndim < 2:
            return None
        if valid.shape[0] != manifest.num_views or valid.shape[1] != manifest.num_bs:
            return None
        return valid
    except (ManifestError, OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def compare_dataset(
    dataset_dir: Path,
    blocker_aabb: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    min_correlation: float = 0.999,
) -> dict:
    """Return per-(view, BS) correlation, amplitude ratio and prediction for one dataset."""
    manifest = load_rf_dataset_manifest(dataset_dir)

    config = manifest.config
    if not np.isclose(
        float(config["vertical_spacing_lambda"]), float(config["horizontal_spacing_lambda"])
    ):
        raise ValueError(
            "vertical_spacing_lambda must equal horizontal_spacing_lambda: "
            "ApertureSpec has a single spacing_lambda"
        )

    bandwidth_hz = float(config["bandwidth_hz"])
    num_bins = len(manifest.frequency_offsets_hz)
    spec = ApertureSpec(
        rows=manifest.rx_rows,
        cols=manifest.rx_cols,
        carrier_hz=manifest.carrier_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        num_freq=num_bins,
        spacing_lambda=float(config["horizontal_spacing_lambda"]),
    )
    model_grid = frequencies_hz(spec)
    stored_grid = np.asarray(manifest.absolute_frequencies_hz, dtype=np.float64)
    tolerance = 1e-3 * (bandwidth_hz / num_bins)
    if float(np.max(np.abs(model_grid - stored_grid))) > tolerance:
        raise ValueError(
            "frequency grid mismatch: model frequencies from ApertureSpec do not match "
            f"manifest absolute_frequencies_hz (tolerance {tolerance:.6g} Hz)"
        )

    prediction_supported = (
        config.get("tx_pattern") == "tr38901" and config.get("polarization") == "V"
    )
    synthetic_array = bool(config.get("synthetic_array", True))

    aperture_cache: dict[str, np.ndarray] = {}
    valid_counts = _load_valid_path_counts(manifest)
    rows = []
    for view, entry in manifest.pairs():
        if view.view_id not in aperture_cache:
            aperture_cache[view.view_id] = manifest.load_aperture_cfr(view)
        aperture = aperture_cache[view.view_id]
        measured = aperture[entry.bs_index].sum(axis=0)
        bs = manifest.base_station(entry.bs_id)
        bs_position = bs.position_m
        bs_look_at = bs.look_at_m
        ue_position = view.position_m
        ue_orientation = view.orientation_rad
        if ue_orientation is None:
            raise ManifestError(f"view {view.view_id!r} has no 'orientation_rad'")
        model = direct_path_cfr(
            spec,
            bs_position,
            ue_position,
            ue_orientation,
            plane_wave=synthetic_array,
        )
        no_energy = float(np.linalg.norm(measured)) == 0.0
        amplitude_ratio = float(np.linalg.norm(measured) / np.linalg.norm(model))
        correlation = _normalised_correlation(measured, model)
        low_correlation = False if no_energy else bool(correlation < min_correlation)

        predicted_tx_gain = None
        polarization_factor = None
        predicted_ratio = None
        ratio_over_predicted = None
        ratio_over_predicted_db = None
        if prediction_supported:
            predicted_tx_gain, polarization_factor, predicted_ratio = direct_path_ratio_prediction(
                bs_position, bs_look_at, ue_position, ue_orientation
            )
            if not no_energy and predicted_ratio != 0.0:
                ratio_over_predicted = amplitude_ratio / predicted_ratio
                ratio_over_predicted_db = 20.0 * float(np.log10(abs(ratio_over_predicted)))

        if no_energy:
            outlier = bool(prediction_supported)
        else:
            ratio_outlier = (
                ratio_over_predicted_db is not None and abs(ratio_over_predicted_db) > 1.0
            )
            outlier = bool(ratio_outlier or low_correlation)

        segment_blocked = None
        if blocker_aabb is not None:
            segment_blocked = segment_intersects_aabb(
                bs_position, ue_position, blocker_aabb[0], blocker_aabb[1]
            )

        num_valid_paths = None
        if valid_counts is not None:
            num_valid_paths = int(np.count_nonzero(valid_counts[view.index, entry.bs_index]))

        rows.append(
            {
                "view_id": view.view_id,
                "bs_id": entry.bs_id,
                "correlation": correlation,
                "low_correlation": low_correlation,
                "amplitude_ratio_measured_over_model": amplitude_ratio,
                "predicted_tx_gain_amplitude": predicted_tx_gain,
                "polarization_factor": polarization_factor,
                "predicted_ratio": predicted_ratio,
                "ratio_over_predicted": ratio_over_predicted,
                "ratio_over_predicted_db": ratio_over_predicted_db,
                "outlier": outlier,
                "no_energy": no_energy,
                "segment_blocked": segment_blocked,
                "num_valid_paths": num_valid_paths,
            }
        )

    correlations = [row["correlation"] for row in rows]
    per_bs = {}
    for bs_id in manifest.bs_ids:
        bs_rows = [row for row in rows if row["bs_id"] == bs_id]
        bs_correlations = [row["correlation"] for row in bs_rows]
        per_bs[bs_id] = {
            "min_correlation": min(bs_correlations) if bs_correlations else None,
            "max_correlation": max(bs_correlations) if bs_correlations else None,
            "outliers": [row["view_id"] for row in bs_rows if row["outlier"]],
        }
    return {
        "dataset": str(dataset_dir),
        "schema_version": manifest.schema_version,
        "bs_ids": list(manifest.bs_ids),
        "prediction_supported": prediction_supported,
        "synthetic_array": synthetic_array,
        "carrier_hz": spec.carrier_hz,
        "bandwidth_hz": spec.bandwidth_hz,
        "num_freq": spec.num_freq,
        "aperture": f"{spec.rows}x{spec.cols}",
        "min_correlation": min(correlations) if correlations else None,
        "max_correlation": max(correlations) if correlations else None,
        "min_correlation_threshold": float(min_correlation),
        "per_bs": per_bs,
        "outliers": [
            {"view_id": row["view_id"], "bs_id": row["bs_id"]} for row in rows if row["outlier"]
        ],
        "pairs": rows,
    }


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


@click.command()
@click.option(
    "--dataset",
    "dataset_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_DATASET,
    show_default=True,
    help="RF-camera multiview output directory.",
)
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the JSON report.",
)
@click.option(
    "--blocker-aabb",
    "blocker_aabb",
    type=float,
    nargs=6,
    default=None,
    help="Optional XMIN YMIN ZMIN XMAX YMAX ZMAX box; flags views whose BS-UE segment is blocked.",
)
@click.option(
    "--min-correlation",
    "min_correlation",
    type=float,
    default=0.999,
    show_default=True,
    help="Correlation threshold below which a pair is flagged as multipath.",
)
def main(
    dataset_dir: Path,
    json_out: Path | None,
    blocker_aabb: tuple[float, ...] | None,
    min_correlation: float,
) -> None:
    """Report direct-path consistency against a Sionna mock dataset."""
    blocker = None
    if blocker_aabb is not None:
        blocker = (
            (float(blocker_aabb[0]), float(blocker_aabb[1]), float(blocker_aabb[2])),
            (float(blocker_aabb[3]), float(blocker_aabb[4]), float(blocker_aabb[5])),
        )

    try:
        report = compare_dataset(dataset_dir, blocker, min_correlation)
    except (ManifestError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    for row in report["pairs"]:
        flags = []
        if row["outlier"]:
            flags.append("OUTLIER")
        if row["low_correlation"]:
            flags.append("LOW_CORR")
        if row["segment_blocked"]:
            flags.append("BLOCKED")
        if row["no_energy"]:
            flags.append("NO_ENERGY")
        suffix = f" {' '.join(flags)}" if flags else ""
        db_text = (
            f"{row['ratio_over_predicted_db']:+.2f}"
            if row["ratio_over_predicted_db"] is not None
            else "-"
        )
        paths_text = str(row["num_valid_paths"]) if row["num_valid_paths"] is not None else "-"
        click.echo(
            f"{row['view_id']} {row['bs_id']}: correlation={row['correlation']:.6f} "
            f"ratio={row['amplitude_ratio_measured_over_model']:.6f} "
            f"predicted={_format(row['predicted_ratio'])} "
            f"ratio/predicted={_format(row['ratio_over_predicted'])} ({db_text} dB){suffix} "
            f"paths={paths_text}"
        )
    for bs_id in report["bs_ids"]:
        per_bs = report["per_bs"][bs_id]
        click.echo(
            f"{bs_id}: correlation range {per_bs['min_correlation']:.6f} .. "
            f"{per_bs['max_correlation']:.6f}, outliers {per_bs['outliers']}"
        )
    click.echo(
        f"correlation range: {report['min_correlation']:.6f} .. {report['max_correlation']:.6f}"
    )
    click.echo(f"outliers: {report['outliers']}")
    if not report["prediction_supported"]:
        click.echo(
            "note: ratio prediction needs tx_pattern 'tr38901' and polarization 'V'; "
            "predicted ratios are None and no outliers can be flagged"
        )
    if json_out is not None:
        json_out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        click.echo(f"wrote {json_out}")


if __name__ == "__main__":
    main()
