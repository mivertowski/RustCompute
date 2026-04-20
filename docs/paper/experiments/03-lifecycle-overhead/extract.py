#!/usr/bin/env python3
"""Parse paper_lifecycle_overhead stdout into CSV.

Stdout: long-form summary (rule, n, mean, sd, p50, p95, p99, ci).
Optional second arg <out_dir>: writes lifecycle.csv with cols
(rule, mean_ns, ci_halfwidth) for figures/lifecycle-cost.tex.
"""
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

PAT = re.compile(r"PAPER_LIFECYCLE\s+rule=(\w+)\s+trial=\d+\s+ns=(\d+)")


def percentile(xs, p):
    if not xs:
        return 0
    return sorted(xs)[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: extract.py <log> [<wide_out_dir>]")
    log = sys.argv[1]
    wide_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    buckets = defaultdict(list)
    with open(log) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                buckets[m.group(1)].append(int(m.group(2)))

    rule_order = ["Spawn", "Activate", "Quiesce", "Terminate", "Restart"]
    print("rule,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns,ci_halfwidth")
    rows = []
    for rule in rule_order:
        xs = buckets.get(rule, [])
        if not xs:
            continue
        n = len(xs); mean = sum(xs) / n
        sd = sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
        ci = 1.96 * (sd / sqrt(n)) if n > 1 else 0.0
        print(f"{rule},{n},{mean:.1f},{sd:.1f},"
              f"{percentile(xs,50)},{percentile(xs,95)},{percentile(xs,99)},{ci:.2f}")
        rows.append((rule, mean, ci))

    if wide_dir is not None:
        wide_dir.mkdir(parents=True, exist_ok=True)
        with open(wide_dir / "lifecycle.csv", "w") as out:
            out.write("rule,mean_ns,ci_halfwidth\n")
            for r in rows:
                out.write(f"{r[0]},{r[1]:.1f},{r[2]:.2f}\n")


if __name__ == "__main__":
    main()
