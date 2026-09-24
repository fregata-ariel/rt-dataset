"""Matplotlib diagnostics for RF-camera direction-cosine images.

Images are drawn with ``origin="lower"`` so that row/column increase toward
local +kz/+ky. matplotlib is imported lazily with the non-interactive Agg
backend so that headless runs (CI, containers) never need a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOCAL_KY_LABEL = "UE-local horizontal direction cosine ky/k"
LOCAL_KZ_LABEL = "UE-local vertical direction cosine kz/k"


@dataclass(frozen=True)
class Marker:
    """A labelled point drawn on a direction-cosine image."""

    ky: float
    kz: float
    marker: str
    label: str


def pyplot() -> Any:
    """Return ``matplotlib.pyplot`` using the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def normalized_power_db(power: np.ndarray, peak: float) -> np.ndarray:
    """Power relative to ``peak`` in dB, floored at -120 dB."""
    return 10.0 * np.log10(np.maximum(power / peak, 1e-12))


def image_extent(horizontal: np.ndarray, vertical: np.ndarray) -> list[float]:
    """Return the imshow extent for the given horizontal/vertical axes."""
    return [
        float(horizontal[0]),
        float(horizontal[-1]),
        float(vertical[0]),
        float(vertical[-1]),
    ]


def save_direction_image(
    image: np.ndarray,
    output_path: Path,
    *,
    extent: list[float],
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    markers: Sequence[Marker] = (),
    xlabel: str = LOCAL_KY_LABEL,
    ylabel: str = LOCAL_KZ_LABEL,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> Path:
    """Save one direction-cosine image (power, phase or delay map) as PNG."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    artist = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    for marker in markers:
        ax.scatter([marker.ky], [marker.kz], marker=marker.marker, label=marker.label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if markers:
        ax.legend()
    fig.colorbar(artist, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
