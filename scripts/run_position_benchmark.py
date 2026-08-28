#!/usr/bin/env python3
"""Dry-run positional benchmark orchestration plans.

Phase 12C2B intentionally builds commands only. It never executes training,
explanation, comparison, or null-baseline stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.position_benchmark_manifest import (
        BenchmarkManifestError,
        build_human_summary,
        build_resolved_plan,
        write_resolved_plan,
    )
else:
    from .position_benchmark_manifest import (
        BenchmarkManifestError,
        build_human_summary,
        build_resolved_plan,
        write_resolved_plan,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the dry-run-only positional benchmark CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Position benchmark manifest YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and print planned commands without execution",
    )
    parser.add_argument(
        "--out-plan",
        type=Path,
        default=None,
        help="Optional resolved-plan YAML path; refused if it already exists",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Python executable path or name to use in planned argv",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Runtime device override for planned train/explain commands",
    )
    parser.add_argument(
        "--allow-existing-outputs",
        action="store_true",
        help="Downgrade non-empty planned output directories to dry-run warnings",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if not args.dry_run:
        print("error: execution is not implemented yet; rerun with --dry-run", file=sys.stderr)
        return 2
    try:
        plan = build_resolved_plan(
            args.manifest,
            python_override=args.python_executable,
            device_override=args.device,
            allow_existing_outputs=args.allow_existing_outputs,
        )
        if args.out_plan is not None:
            write_resolved_plan(args.out_plan, plan)
        print(build_human_summary(plan))
    except BenchmarkManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
