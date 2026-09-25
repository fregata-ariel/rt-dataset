"""CI: run and evaluate the heavy tomography smoke (#15 T22).

Runs :func:`plateau_rt.application.rf_tomography_smoke.run_smoke` on a dataset
with tomography GT, evaluates the §6.7 gates and writes the smoke report.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from plateau_rt.application.rf_tomography_smoke import (
    SMOKE_MAX_RUNTIME_S,
    SMOKE_NUM_BINS,
    SMOKE_WORKERS,
    evaluate_smoke,
    exit_code,
    run_smoke,
    write_smoke_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run and/or evaluate the smoke output; return the gate exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="tomography dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="smoke output directory")
    parser.add_argument("--workers", type=int, default=SMOKE_WORKERS)
    parser.add_argument("--num-bins", type=int, default=SMOKE_NUM_BINS)
    parser.add_argument("--max-runtime-s", type=float, default=SMOKE_MAX_RUNTIME_S)
    parser.add_argument("--check-only", action="store_true", help="evaluate an existing output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        if not args.check_only:
            run_smoke(
                args.dataset,
                args.out,
                workers=args.workers,
                num_bins=args.num_bins,
                overwrite=args.overwrite,
            )
        checks = evaluate_smoke(args.out, dataset=args.dataset, max_runtime_s=args.max_runtime_s)
        summary_path, report_path = write_smoke_report(args.out, checks)
    except (ValueError, FileExistsError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for check in checks:
        print(f"{check.status.upper()} {check.name} {check.measured}")
    print(f"report: {report_path} summary: {summary_path}")
    return exit_code(checks)


if __name__ == "__main__":
    sys.exit(main())
