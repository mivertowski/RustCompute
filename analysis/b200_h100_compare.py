#!/usr/bin/env python3
"""Compare H100 baseline Criterion estimates against B200 results.

Emits a markdown table with mean / 95% CI / speedup per benchmark.
Matplotlib plots are best-effort (no hard dependency).

Spec: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, Optional, Tuple


def find_estimates(root: pathlib.Path) -> Dict[str, dict]:
    """Walk a Criterion output tree and collect benchmark name -> estimates dict."""
    out: Dict[str, dict] = {}
    if not root.exists():
        return out
    for path in root.rglob("estimates.json"):
        # Criterion layout: <root>/<bench_id>/[<sub>/]new/estimates.json
        rel = path.relative_to(root)
        parts = rel.parts
        if "new" not in parts:
            continue
        new_idx = parts.index("new")
        if new_idx == 0:
            continue
        bench_id = "/".join(parts[:new_idx])
        try:
            with path.open() as fh:
                out[bench_id] = json.load(fh)
        except json.JSONDecodeError:
            print(f"warn: malformed json: {path}", file=sys.stderr)
    return out


def fmt_ns(ns: float) -> str:
    """Format a nanosecond duration with a sensible unit."""
    if ns < 1_000:
        return f"{ns:.2f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.3f} s"


def speedup(h100_ns: float, b200_ns: float) -> str:
    if b200_ns <= 0:
        return "n/a"
    ratio = h100_ns / b200_ns
    return f"{ratio:.2f}×"


def mean_ci(est: dict) -> Optional[Tuple[float, float, float]]:
    """Return (mean_ns, low_ns, high_ns) from a Criterion estimates dict."""
    mean = est.get("mean", {})
    pe = mean.get("point_estimate")
    ci = mean.get("confidence_interval", {})
    lo = ci.get("lower_bound")
    hi = ci.get("upper_bound")
    if pe is None or lo is None or hi is None:
        return None
    return float(pe), float(lo), float(hi)


def build_table(h100: Dict[str, dict], b200: Dict[str, dict]) -> str:
    keys = sorted(set(h100) | set(b200))
    lines = [
        "| Benchmark | H100 mean (95% CI) | B200 mean (95% CI) | B200 / H100 |",
        "|---|---|---|---|",
    ]
    for k in keys:
        h = mean_ci(h100.get(k, {})) if k in h100 else None
        b = mean_ci(b200.get(k, {})) if k in b200 else None
        if h is None and b is None:
            continue
        h_cell = (
            f"{fmt_ns(h[0])} ({fmt_ns(h[1])}, {fmt_ns(h[2])})" if h else "—"
        )
        b_cell = (
            f"{fmt_ns(b[0])} ({fmt_ns(b[1])}, {fmt_ns(b[2])})" if b else "—"
        )
        sp = speedup(h[0], b[0]) if (h and b) else "—"
        lines.append(f"| `{k}` | {h_cell} | {b_cell} | {sp} |")
    return "\n".join(lines) + "\n"


def maybe_plot(
    h100: Dict[str, dict], b200: Dict[str, dict], out_dir: pathlib.Path
) -> None:
    """Emit a simple bar plot per benchmark if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("info: matplotlib unavailable — skipping plots", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(set(h100) & set(b200)):
        h = mean_ci(h100[k])
        b = mean_ci(b200[k])
        if h is None or b is None:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        labels = ["H100", "B200"]
        means = [h[0], b[0]]
        errs = [
            [h[0] - h[1], b[0] - b[1]],  # lower error
            [h[2] - h[0], b[2] - b[0]],  # upper error
        ]
        ax.bar(labels, means, yerr=errs, capsize=4)
        ax.set_ylabel("ns (lower is better)")
        ax.set_title(k)
        fig.tight_layout()
        safe = k.replace("/", "__").replace(" ", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=120)
        plt.close(fig)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", required=True, type=pathlib.Path)
    parser.add_argument("--b200", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(list(argv))

    h100 = find_estimates(args.h100)
    b200 = find_estimates(args.b200)
    if not h100 and not b200:
        print("error: neither --h100 nor --b200 contained estimates.json", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_table(h100, b200))
    maybe_plot(h100, b200, args.out.parent / "plots")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
