#!/usr/bin/env python3
"""Generate demo metrics for the compact CNN path."""

from __future__ import annotations

import argparse
import json
import pathlib

from qml_cnn_welds.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1] / "configs/sample_config.yaml"),
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results" / "sample_metrics.json"),
        help="Metrics JSON output path.",
    )
    args = parser.parse_args()

    metrics = train_from_config(args.config)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
