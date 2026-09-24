#!/usr/bin/env python3
"""Plan (dry-run) or execute positional benchmark orchestration plans.

Phase 12C2B dry-run planning builds commands only and never executes a
stage. Phase 12C3B1 adds schema_version 2 manifests, which additionally plan
one null-trained model per positional strategy on one shared null artifact
and the per-strategy bootstrap calibration command.

Phase 12C3B2B adds execution of one reviewed schema-v2 plan::

    run_position_benchmark.py MANIFEST --dry-run [--out-plan PLAN] [--python ...]
        [--device ...] [--allow-existing-outputs]
    run_position_benchmark.py MANIFEST --execute-plan PLAN [--resume]

Execution runs the exact planned argv lists through the paired stage DAG
behind the full null preflight (see ``position_benchmark_execution``).
Planning options are rejected with ``--execute-plan``: the reviewed plan is
the only execution authority. A production plan must be generated at the
exact executor revision that will run it; plans from earlier revisions are
development fixtures and fail the repository gate. Schema-v1 plans are never
executable. Exit codes: 0 success, 2 ordinary planning/execution error
(concise message, no traceback), 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.position_benchmark_execution import (
        ExecutionFoundationError,
        execute_benchmark_plan,
    )
    from scripts.position_benchmark_manifest import (
        BenchmarkManifestError,
        build_human_summary,
        build_resolved_plan,
        write_resolved_plan,
    )
    from scripts.position_benchmark_records import StageRecordError
else:
    from .position_benchmark_execution import (
        ExecutionFoundationError,
        execute_benchmark_plan,
    )
    from .position_benchmark_manifest import (
        BenchmarkManifestError,
        build_human_summary,
        build_resolved_plan,
        write_resolved_plan,
    )
    from .position_benchmark_records import StageRecordError

INTERRUPTED_EXIT_CODE = 130
USAGE_EXIT_CODE = 2


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the positional benchmark dry-run / execution CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Position benchmark manifest YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and print planned commands without execution",
    )
    parser.add_argument(
        "--execute-plan",
        type=Path,
        default=None,
        metavar="PLAN",
        help="Execute one reviewed schema-v2 resolved plan (mutually exclusive with --dry-run)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --execute-plan: revalidate completed stages and continue the DAG",
    )
    parser.add_argument(
        "--out-plan",
        type=Path,
        default=None,
        help="Dry-run only: resolved-plan YAML path; refused if it already exists",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Dry-run only: Python executable path or name to use in planned argv",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Dry-run only: runtime device override for planned train/explain commands",
    )
    parser.add_argument(
        "--allow-existing-outputs",
        action="store_true",
        help="Dry-run only: downgrade non-empty planned output directories to warnings",
    )
    return parser


def _usage_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return USAGE_EXIT_CODE


def _dry_run(args: argparse.Namespace) -> int:
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
        return USAGE_EXIT_CODE
    return 0


def format_execution_summary(result: dict) -> str:
    """Return the concise human-readable execution summary printed on success."""
    summary = result["summary"]
    return "\n".join(
        [
            "POSITION BENCHMARK EXECUTION COMPLETE (raw paired benchmark)",
            f"Resolved plan SHA256: {summary['resolved_plan_sha256']}",
            f"Repository revision: {summary['repository_revision']}",
            f"Runs: {summary['n_runs']}",
            (
                f"Stages completed: {summary['completed_stages']}/{summary['n_stages']} "
                f"(executed {len(result['executed'])}, reused {len(result['reused'])})"
            ),
            (
                "Real/null training completed: "
                f"{summary['real_training_completed']}/{summary['null_training_completed']}"
            ),
            (
                "Real/null explanations completed: "
                f"{summary['real_explanations_completed']}/"
                f"{summary['null_explanations_completed']}"
            ),
            f"Pair validations completed: {summary['pair_validations_completed']}",
            f"Calibrations completed: {summary['calibrations_completed']}",
            f"Shared-null validation: {summary['shared_null_validation']}",
            "Raw comparisons completed: " + ", ".join(summary["raw_comparisons_completed"]),
            (
                "Calibrated cross-strategy ranking comparison: NOT executed "
                "(gate closed until Phase 12C3C)"
            ),
            f"Summary: {result['summary_path']}",
        ]
    )


def _execute(args: argparse.Namespace) -> int:
    rejected = [
        flag
        for flag, present in (
            ("--python", args.python_executable is not None),
            ("--device", args.device is not None),
            ("--allow-existing-outputs", args.allow_existing_outputs),
            ("--out-plan", args.out_plan is not None),
        )
        if present
    ]
    if rejected:
        return _usage_error(
            f"{', '.join(rejected)} cannot be combined with --execute-plan; the reviewed "
            "plan is the only execution authority (regenerate it with --dry-run instead)"
        )
    try:
        result = execute_benchmark_plan(
            args.execute_plan, manifest_path=args.manifest, resume=args.resume
        )
    except KeyboardInterrupt:
        print("error: benchmark execution interrupted", file=sys.stderr)
        return INTERRUPTED_EXIT_CODE
    except (ExecutionFoundationError, StageRecordError, BenchmarkManifestError) as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_EXIT_CODE
    print(format_execution_summary(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.dry_run and args.execute_plan is not None:
        return _usage_error("--dry-run and --execute-plan are mutually exclusive")
    if args.resume and args.execute_plan is None:
        return _usage_error("--resume requires --execute-plan PLAN")
    if args.execute_plan is not None:
        return _execute(args)
    if not args.dry_run:
        return _usage_error("choose --dry-run or --execute-plan PLAN")
    return _dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
