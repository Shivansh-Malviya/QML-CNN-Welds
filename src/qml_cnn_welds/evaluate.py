"""Evaluation entry point for configured runs."""

from __future__ import annotations

from .train import train_from_config


def evaluate_from_config(config_path: str) -> dict[str, float]:
    """Run the configured training/evaluation path and return metrics."""
    return train_from_config(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a configured model.")
    parser.add_argument("config", type=str, help="YAML configuration path.")
    args = parser.parse_args()
    metrics = evaluate_from_config(args.config)
    print(metrics)
