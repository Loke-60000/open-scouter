"""
Convert an assistant-axis ``.pt`` to ``.npz`` (numpy) so serving needs no torch.

Run this once (torch required), then point ``SCOUTER_AXIS`` at the ``.npz``.

Usage:
    python scripts/convert_axis.py --in hf:qwen-3-32b --out qwen3_axis.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from scouter.axis import load_axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, help="axis .pt path or 'hf:<key>'")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--hf-repo", default="lu-christina/assistant-axis-vectors")
    args = ap.parse_args()

    axis = load_axis(args.src, hf_repo=args.hf_repo)
    np.savez(args.out, axis=axis.directions)
    print(f"wrote {args.out}: shape {axis.directions.shape}")


if __name__ == "__main__":
    main()
