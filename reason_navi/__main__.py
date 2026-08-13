"""Command line entrypoint for the two-stage ObjectNav runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    # Preserve direct ``python reason_navi/__main__.py`` execution while
    # keeping path mutation confined to the CLI boundary.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reason_navi.navigation.config import (
    load_json_object,
    load_navigation_config,
)
from reason_navi.navigation.runner import run_navigation


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical two-stage ObjectNav navigation runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.json"),
        help="Navigation runtime configuration JSON",
    )
    parser.add_argument(
        "--request",
        "--actions",
        dest="request",
        required=True,
        help="Navigation request JSON (historically named action.json)",
    )
    parser.add_argument(
        "--wall-mask",
        default=None,
        help=(
            "Optional wall-mask override; requires --map-metadata"
        ),
    )
    parser.add_argument(
        "--map-metadata",
        default=None,
        help="Projection metadata paired with --wall-mask",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config.output_dir",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a traceback on failure"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    try:
        config = load_navigation_config(args.config)
        request = load_json_object(args.request)
        result = run_navigation(
            config,
            request,
            wall_mask=(
                str(Path(args.wall_mask).expanduser().resolve())
                if args.wall_mask
                else None
            ),
            map_metadata=(
                str(Path(args.map_metadata).expanduser().resolve())
                if args.map_metadata
                else None
            ),
            output_dir=(
                str(Path(args.output_dir).expanduser().resolve())
                if args.output_dir
                else None
            ),
            request_base_dir=Path(args.request).expanduser().resolve().parent,
        )
        execution = result.report["execution_info"]
        print("Navigation completed")
        print(
            f"Frames: {execution['total_frames']}; "
            f"duration: {execution['video_duration_seconds']:.2f}s"
        )
        print(f"Video: {result.video_path}")
        print(f"Report: {result.report_path}")
        return 0
    except KeyboardInterrupt:
        print("Navigation interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Navigation failed: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
