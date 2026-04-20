#!/usr/bin/env python3
"""Parse paper_nvlink_migration stdout into CSV.

Stdout: long-form
Optional second arg <out_dir>: writes migration_p2p.csv and migration_host.csv
(format expected by figures/migration-cost.tex).
"""
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

PAT = re.compile(r"PAPER_MIGRATION\s+path=(\w+)\s+size=(\d+)\s+trial=\d+\s+ns=(\d+)")


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
                buckets[(m.group(1), int(m.group(2)))].append(int(m.group(3)))

    rows = []
    path_order = {"p2p": 0, "host": 1}
    print("path,size_bytes,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns,ci_halfwidth")
    for (path, size), xs in sorted(
        buckets.items(), key=lambda kv: (path_order.get(kv[0][0], 99), kv[0][1])
    ):
        n = len(xs); mean = sum(xs) / n
        sd = sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
        ci = 1.96 * (sd / sqrt(n)) if n > 1 else 0.0
        print(f"{path},{size},{n},{mean:.1f},{sd:.1f},"
              f"{percentile(xs,50)},{percentile(xs,95)},{percentile(xs,99)},{ci:.2f}")
        rows.append((path, size, mean, ci))

    if wide_dir is not None:
        wide_dir.mkdir(parents=True, exist_ok=True)
        for path in ("p2p", "host"):
            with open(wide_dir / f"migration_{path}.csv", "w") as out:
                out.write("size_bytes,mean_ns,ci_halfwidth\n")
                for r in rows:
                    if r[0] == path:
                        out.write(f"{r[1]},{r[2]:.1f},{r[3]:.2f}\n")


if __name__ == "__main__":
    main()
