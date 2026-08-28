"""1 BS / 1 UE RF-camera MVP built on Sionna RT.

The raw observation is a complex channel frequency response (CFR) sampled
across a planar receive aperture. A simple 2-D spatial FFT is used as the
first "development" step to produce a human-inspectable angular-spectrum
image. This module intentionally keeps the RF-camera image formation separate
from Sionna's optical scene renderer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene


@dataclass(frozen=True)
class RFCameraConfig:
    """Configuration for the first 1-BS / 1-UE RF-camera milestone."""

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_frequency_bins: int = 64

    # Keep Tx spatially simple for the first milestone. The RF-camera aperture
    # is the Rx array; Tx beamforming will be introduced independently later.
    tx_position: tuple[float, float, float] = (-50.0, -50.0, 30.0)
    tx_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ue_position: tuple[float, float, float] = (0.0, 0.0, 1.5)
    ue_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    rx_rows: int = 8
    rx_cols: int = 8
    vertical_spacing_lambda: float = 0.5
    horizontal_spacing_lambda: float = 0.5

    max_depth: int = 5
    synthetic_array: bool = True
    seed: int = 42

    # Display only. Zero-padding improves readability of the FFT image but does
    # not increase the physical angular resolution of the aperture.
    fft_rows: int = 128
    fft_cols: int = 128

    def validate(self) -> None:
        if self.num_frequency_bins < 1:
            raise ValueError("num_frequency_bins must be >= 1")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be > 0")
        if self.rx_rows < 1 or self.rx_cols < 1:
            raise ValueError("rx_rows and rx_cols must be >= 1")
        if self.fft_rows < self.rx_rows or self.fft_cols < self.rx_cols:
            raise ValueError("FFT grid must not be smaller than the receive aperture")


@dataclass(frozen=True)
class RFCameraArtifacts:
    aperture_cfr: Path
    angular_cfr: Path
    metadata: Path
    power_png: Path
    phase_png: Path
    path_gt: Path


class RFCameraMVP:
    """Generate one coherent RF-camera view from one BS and one UE aperture."""

    def __init__(self, xml_path: Path, config: RFCameraConfig | None = None):
        self.xml_path = Path(xml_path)
        self.config = config or RFCameraConfig()
        self.config.validate()

    def run(self, output_dir: Path) -> RFCameraArtifacts:
        """Trace paths, export aperture CFR, and develop an FFT RF image."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        scene = load_scene(str(self.xml_path))
        scene.frequency = cfg.carrier_frequency_hz

        # For the first milestone we excite one Tx antenna/port. This keeps
        # transmit beamforming out of the RF-camera image-formation problem.
        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )
        scene.rx_array = PlanarArray(
            num_rows=cfg.rx_rows,
            num_cols=cfg.rx_cols,
            vertical_spacing=cfg.vertical_spacing_lambda,
            horizontal_spacing=cfg.horizontal_spacing_lambda,
            pattern="dipole",
            polarization="V",
        )

        tx = Transmitter(
            name="rf_camera_bs_0",
            position=list(cfg.tx_position),
            orientation=list(cfg.tx_orientation),
        )
        rx = Receiver(
            name="rf_camera_ue_0",
            position=list(cfg.ue_position),
            orientation=list(cfg.ue_orientation),
        )
        scene.add(tx)
        scene.add(rx)

        print("=== RF Camera MVP: path tracing ===")
        print(f"scene={self.xml_path}")
        print(f"carrier={cfg.carrier_frequency_hz / 1e9:.6f} GHz")
        print(f"BS position={cfg.tx_position}")
        print(f"UE position={cfg.ue_position}, orientation={cfg.ue_orientation} rad")
        print(
            "Rx aperture="
            f"{cfg.rx_rows}x{cfg.rx_cols}, spacing="
            f"({cfg.vertical_spacing_lambda}, {cfg.horizontal_spacing_lambda}) lambda"
        )

        solver = PathSolver()
        paths = solver(
            scene=scene,
            max_depth=cfg.max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
        )

        # Paths.cfr() operates on baseband frequency offsets around the scene's
        # carrier. Keeping the carrier in scene.frequency avoids applying the
        # carrier propagation phase a second time.
        frequency_offsets_hz = _frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
        cfr = np.asarray(
            paths.cfr(
                frequencies=frequency_offsets_hz,
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
        )

        print(f"Paths.cfr shape={cfr.shape}, dtype={cfr.dtype}")
        expected_rx_ant = cfg.rx_rows * cfg.rx_cols
        expected_shape = (1, expected_rx_ant, 1, 1, 1, cfg.num_frequency_bins)
        if cfr.shape != expected_shape:
            raise RuntimeError(
                "Unexpected Sionna Paths.cfr shape. "
                f"expected={expected_shape}, actual={cfr.shape}. "
                "Please share this log; the extractor should be adjusted before continuing."
            )

        # [num_rx, num_rx_ant, num_tx, num_tx_ant, time, frequency]
        aperture_flat = cfr[0, :, 0, 0, 0, :]
        aperture_cfr = _reshape_planar_column_first(
            aperture_flat,
            rows=cfg.rx_rows,
            cols=cfg.rx_cols,
        )
        # Shape: [row, col, frequency]
        print(f"aperture_cfr shape={aperture_cfr.shape}, dtype={aperture_cfr.dtype}")

        angular_cfr = aperture_to_angular_fft(
            aperture_cfr,
            fft_rows=cfg.fft_rows,
            fft_cols=cfg.fft_cols,
        )
        print(f"angular_cfr shape={angular_cfr.shape}, dtype={angular_cfr.dtype}")

        aperture_path = output_dir / "aperture_cfr.npy"
        angular_path = output_dir / "angular_cfr.npy"
        np.save(aperture_path, aperture_cfr.astype(np.complex64, copy=False))
        np.save(angular_path, angular_cfr.astype(np.complex64, copy=False))

        path_gt_path = output_dir / "path_gt.npz"
        self._save_path_gt(paths, path_gt_path)

        center_bin = cfg.num_frequency_bins // 2
        power_png = output_dir / "angular_power_center.png"
        phase_png = output_dir / "angular_phase_center.png"
        _render_debug_images(
            angular_cfr[:, :, center_bin],
            power_png=power_png,
            phase_png=phase_png,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        )

        metadata_path = output_dir / "rf_camera_metadata.json"
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "mode": "1bs_1ue_rf_camera_mvp",
            "config": asdict(cfg),
            "frequency_offsets_hz": frequency_offsets_hz.tolist(),
            "absolute_frequencies_hz": (
                cfg.carrier_frequency_hz + frequency_offsets_hz
            ).tolist(),
            "sionna_cfr_axis_order": [
                "rx",
                "rx_ant",
                "tx",
                "tx_ant",
                "time",
                "frequency_offset",
            ],
            "sionna_cfr_shape": list(cfr.shape),
            "aperture_axis_order": ["row", "col", "frequency_offset"],
            "aperture_shape": list(aperture_cfr.shape),
            "planar_array_numbering": "column-first, top-left to bottom-right",
            "array_plane": "local y-z",
            "angular_fft_axis_order": ["vertical_spatial_frequency", "horizontal_spatial_frequency", "frequency_offset"],
            "angular_fft_shape": list(angular_cfr.shape),
            "notes": [
                "angular_cfr is a zero-padded 2-D spatial FFT diagnostic, not yet a calibrated AoA image",
                "Tx uses one active antenna/port for the MVP",
                "ideal coherent Sionna phase is preserved",
                "absolute path delays are requested with normalize_delays=False",
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("=== RF Camera MVP complete ===")
        for path in (
            aperture_path,
            angular_path,
            path_gt_path,
            power_png,
            phase_png,
            metadata_path,
        ):
            print(f"  {path}")

        return RFCameraArtifacts(
            aperture_cfr=aperture_path,
            angular_cfr=angular_path,
            metadata=metadata_path,
            power_png=power_png,
            phase_png=phase_png,
            path_gt=path_gt_path,
        )

    @staticmethod
    def _save_path_gt(paths: Any, output_path: Path) -> None:
        """Save enough path GT to diagnose CFR/image formation in the MVP."""
        # cir() returns complex coefficients and delays. Keep absolute delays.
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        payload: dict[str, np.ndarray] = {
            "a": np.asarray(a),
            "tau": np.asarray(tau),
        }
        for name in ("valid", "theta_t", "phi_t", "theta_r", "phi_r"):
            try:
                payload[name] = np.asarray(getattr(paths, name))
            except Exception as exc:  # pragma: no cover - depends on Sionna backend
                print(f"Warning: could not export paths.{name}: {exc}")
        np.savez_compressed(output_path, **payload)


def _frequency_offsets(bandwidth_hz: float, num_bins: int) -> np.ndarray:
    """Return evenly spaced baseband offsets centered on DC."""
    if num_bins == 1:
        return np.array([0.0], dtype=np.float32)
    # Endpoint=False gives a regular FFT/OFDM-like grid centered around DC.
    spacing = bandwidth_hz / num_bins
    indices = np.arange(num_bins, dtype=np.float64) - num_bins // 2
    return (indices * spacing).astype(np.float32)


def _reshape_planar_column_first(
    aperture_flat: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Restore Sionna PlanarArray's column-first antenna numbering.

    `aperture_flat` has shape [rx_ant, ...]. Antenna indices walk down all
    rows of the first column before moving to the next column.
    """
    aperture_flat = np.asarray(aperture_flat)
    if aperture_flat.shape[0] != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} antenna samples, got {aperture_flat.shape[0]}"
        )

    out = np.empty((rows, cols) + aperture_flat.shape[1:], dtype=aperture_flat.dtype)
    for antenna_index in range(rows * cols):
        row = antenna_index % rows
        col = antenna_index // rows
        out[row, col, ...] = aperture_flat[antenna_index, ...]
    return out


def aperture_to_angular_fft(
    aperture_cfr: np.ndarray,
    *,
    fft_rows: int,
    fft_cols: int,
) -> np.ndarray:
    """Create a first-look angular spectrum by spatial FFT of the UE aperture.

    The receive aperture is the local y-z plane. The returned image is kept in
    spatial-frequency coordinates for this milestone; conversion to calibrated
    AoA angles belongs to the next image-formation step.
    """
    aperture_cfr = np.asarray(aperture_cfr)
    if aperture_cfr.ndim != 3:
        raise ValueError("aperture_cfr must have shape [row, col, frequency]")
    if fft_rows < aperture_cfr.shape[0] or fft_cols < aperture_cfr.shape[1]:
        raise ValueError("FFT grid must not be smaller than aperture")

    spectrum = np.fft.fft2(aperture_cfr, s=(fft_rows, fft_cols), axes=(0, 1))
    return np.fft.fftshift(spectrum, axes=(0, 1))


def _render_debug_images(
    angular_slice: np.ndarray,
    *,
    power_png: Path,
    phase_png: Path,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> None:
    """Render log-power and phase diagnostics for one frequency bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power = np.abs(angular_slice) ** 2
    peak = float(np.max(power)) if power.size else 0.0
    power_db = 10.0 * np.log10(np.maximum(power / max(peak, 1e-30), 1e-12))
    phase = np.angle(angular_slice)

    # fftfreq values are cycles/sample. Dividing by d/lambda converts them to
    # direction-cosine-like spatial coordinates k_axis/k for the planar array.
    horizontal = np.fft.fftshift(np.fft.fftfreq(angular_slice.shape[1]))
    vertical = np.fft.fftshift(np.fft.fftfreq(angular_slice.shape[0]))
    horizontal = horizontal / horizontal_spacing_lambda
    vertical = vertical / vertical_spacing_lambda
    extent = [horizontal[0], horizontal[-1], vertical[0], vertical[-1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        power_db,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-60.0,
        vmax=0.0,
    )
    ax.set_xlabel("horizontal spatial coordinate ky/k")
    ax.set_ylabel("vertical spatial coordinate kz/k")
    ax.set_title("RF camera angular spectrum: normalized power [dB]")
    fig.colorbar(image, ax=ax, label="dB relative to peak")
    fig.tight_layout()
    fig.savefig(power_png, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        phase,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.set_xlabel("horizontal spatial coordinate ky/k")
    ax.set_ylabel("vertical spatial coordinate kz/k")
    ax.set_title("RF camera angular spectrum: phase [rad]")
    fig.colorbar(image, ax=ax, label="phase [rad]")
    fig.tight_layout()
    fig.savefig(phase_png, dpi=150)
    plt.close(fig)
