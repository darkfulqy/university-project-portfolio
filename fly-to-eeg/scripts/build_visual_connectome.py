#!/usr/bin/env python3
"""Build the independent whole-traced-CNS visual-response graph."""
from pathlib import Path
import argparse
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from flyvision.connectome import build_full_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/male_cns_v1_0_visual_full")
    args = parser.parse_args()
    build_full_graph(PROJECT, args.output, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
