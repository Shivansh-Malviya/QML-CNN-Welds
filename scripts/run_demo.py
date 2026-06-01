#!/usr/bin/env python3
"""Run the compact CNN demonstration path."""

from __future__ import annotations

import argparse
import pathlib

from qml_cnn_welds.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact training session.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1] / "configs/sample_config.yaml"),
        help="YAML configuration path.",
    )
    args = parser.parse_args()
    metrics = train_from_config(args.config)
    print("\nFinal metrics:\n", metrics)


if __name__ == "__main__":
    main()
