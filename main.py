"""CLI entry point.

Usage:
    uv run python main.py
    uv run python main.py --manuscript data/manuscript.pdf --reviews data/reviews.txt --out outputs/rebuttal.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from response_agent.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft a manuscript rebuttal from reviewer comments.")
    parser.add_argument("--manuscript", type=Path, default=Path("data/manuscript.pdf"))
    parser.add_argument("--reviews", type=Path, default=Path("data/reviews.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rebuttal.md"))
    parser.add_argument("--top-k", type=int, default=4, help="Manuscript chunks to retrieve per comment.")
    parser.add_argument("--refine-passes", type=int, default=1, help="Max Critic-driven refine passes.")
    parser.add_argument("--adversarial", action="store_true",
                        help="Also generate a hostile-reviewer follow-up per comment.")
    args = parser.parse_args()

    out = run(
        manuscript_path=args.manuscript,
        reviews_path=args.reviews,
        output_path=args.out,
        top_k=args.top_k,
        max_refine_passes=args.refine_passes,
        adversarial=args.adversarial,
    )
    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
