#!/usr/bin/env python3
"""Check whether the top-weighted MultiNest samples reproduce the stored likelihoods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import analyse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute likelihoods for the top-weighted MultiNest samples."
    )
    parser.add_argument(
        "--model",
        choices=("clear", "hazy"),
        default="clear",
        help="Which retrieval output to inspect.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top-weighted rows to print.",
    )
    args = parser.parse_args()

    try:
        import retrieval
    except ImportError as exc:
        raise SystemExit(
            "Failed to import retrieval.py. Run this script in the same environment "
            "that your retrieval uses (the one with picaso/photochem installed)."
        ) from exc

    name = args.model
    num_params = len(retrieval.PARAM_NAMES[name])
    output_prefix = f"pymultinest/{name}/{name}"

    analyzer = analyse.Analyzer(
        n_params=num_params,
        outputfiles_basename=output_prefix,
        verbose=False,
    )
    data = analyzer.get_data()

    weights = np.asarray(data[:, 0], dtype=float)
    idx = np.argsort(weights)[::-1][: args.top]

    print(f"Inspecting top {len(idx)} weighted samples for model '{name}'")
    print(f"clear.txt shape = {data.shape}")
    print()

    for rank, i in enumerate(idx, start=1):
        row = data[i]
        w = float(row[0])
        neg2loglike = float(row[1])
        x = np.asarray(row[2:], dtype=float)
        ll_recomputed = float(retrieval.loglike(x, name))
        neg2loglike_recomputed = -2.0 * ll_recomputed
        print(f"rank {rank:2d} row {i:6d}")
        print(f"  stored weight              = {w:.18e}")
        print(f"  stored -2 loglike          = {neg2loglike:.18e}")
        print(f"  recomputed loglike         = {ll_recomputed:.18e}")
        print(f"  recomputed -2 loglike      = {neg2loglike_recomputed:.18e}")
        print(f"  abs diff in -2 loglike     = {abs(neg2loglike - neg2loglike_recomputed):.18e}")
        print()


if __name__ == "__main__":
    main()
