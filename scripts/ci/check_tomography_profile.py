"""CI: verify a tomography dataset profile dataset (#15 T20).

Recomputes the ``tomography`` manifest section, the pose bank and the oracle
``los=False`` trace contract through
:func:`plateau_rt.application.rf_tomography_profile.verify_tomography_dataset`
and additionally checks the profile name, the N = 128 frequency grid and the
view / BS counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.application.rf_tomography_profile import (
    load_tomography_section,
    verify_tomography_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one tomography dataset directory; return 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="tomography dataset directory")
    parser.add_argument("--profile", required=True, help="expected tomography profile name")
    parser.add_argument("--num-views", type=int, default=None, help="expected view count")
    parser.add_argument("--num-bs", type=int, default=None, help="expected BS count")
    parser.add_argument("--oracle-rtol", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        summary = verify_tomography_dataset(args.dataset, oracle_rtol=args.oracle_rtol)
        section = load_tomography_section(args.dataset)
        manifest = load_rf_dataset_manifest(args.dataset)
        if section["profile"] != args.profile:
            raise ValueError(
                f"tomography profile is {section['profile']!r}, expected {args.profile!r}"
            )
        if manifest.num_frequency_bins != 128:
            raise ValueError(f"num_frequency_bins is {manifest.num_frequency_bins}, expected 128")
        if args.num_views is not None and manifest.num_views != args.num_views:
            raise ValueError(f"num_views is {manifest.num_views}, expected {args.num_views}")
        if args.num_bs is not None and manifest.num_bs != args.num_bs:
            raise ValueError(f"num_bs is {manifest.num_bs}, expected {args.num_bs}")
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
