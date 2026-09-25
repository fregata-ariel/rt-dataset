"""CI: verify T21 tomography resynthesis, BS pattern and polarisation checks.

Checks L0f VS resynthesis NMSE < 1e-3 raw and gauge-aligned, the operator's
first-order departure model (order-0/1 gain ratio within 0.01 dB), tr38901
direct-path ratios within 0.5 dB with the V-pol model, reports eps_LoS and
polarisation statistics, and runs two controls (element-row flip and the iso
pattern).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from plateau_rt.application.rf_dataset_manifest import ManifestError
from plateau_rt.application.rf_tomography_resynthesis import (
    DIRECT_PATH_DB_MAX,
    NMSE_MAX,
    resynthesis_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one dataset's resynthesis report; return 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="tomography dataset directory")
    parser.add_argument("--gt", type=Path, default=None, help="explicit tomography GT npz")
    parser.add_argument("--nmse-max", type=float, default=NMSE_MAX, help="L0f NMSE gate")
    parser.add_argument(
        "--direct-db-max",
        type=float,
        default=DIRECT_PATH_DB_MAX,
        help="direct-path ratio gate in dB",
    )
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    try:
        report = resynthesis_report(
            args.dataset,
            gt_file=args.gt,
            nmse_max=args.nmse_max,
            direct_db_max=args.direct_db_max,
        )
    except (ValueError, ManifestError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
