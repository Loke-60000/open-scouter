"""
Generate an assistant-axis vector file from a HuggingFace model.

Thin wrapper around ``scouter.axis.generate`` (also installed as the
``scouter-generate`` console command).

Examples:
    # Compact built-in personas (fast, good for small models):
    python scripts/generate_axis.py --model Qwen/Qwen3-0.6B --out qwen3_axis.npz

    # Richer axis using assistant-axis's 275-role list:
    git submodule update --init third_party/assistant-axis
    python scripts/generate_axis.py --model Qwen/Qwen3-0.6B --out qwen3_axis.pt \
        --roles submodule --position mean
"""

from scouter.axis.generate import main

if __name__ == "__main__":
    main()
