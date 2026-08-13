"""CLI for the two-stage preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-stage ObjectNav target and portable navigation "
            "artifacts"
        )
    )
    parser.add_argument("config", type=Path, help="Preprocessing config JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    # Delay Habitat/OpenCV imports so ``--help`` works in a lightweight
    # checkout and configuration errors remain easy to diagnose.
    from .pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(str(args.config.expanduser().resolve()))
    return 0 if pipeline.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
