#!/usr/bin/env python3
"""Extract the local v1.0 front-leg graph; never downloads or runs simulations."""
from pathlib import Path
import argparse
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from flycue.connectome import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "data/processed/male_cns_v1_0_cue_leg")
    parser.add_argument("--min-synapses", type=int, default=5)
    parser.add_argument("--strict-known-nt-motors", action="store_true",
                        help="Exclude motor neurons with unknown transmitter instead of retaining terminal readouts")
    args = parser.parse_args()
    build_graph(PROJECT, args.output, args.min_synapses,
                progress=lambda message: print(message, flush=True),
                include_unknown_motor_readouts=not args.strict_known_nt_motors)


if __name__ == "__main__":
    main()
