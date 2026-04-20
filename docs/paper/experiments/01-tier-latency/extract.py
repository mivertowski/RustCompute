#!/usr/bin/env python3
"""Parse paper_tier_latency stdout into CSV with per-(tier,payload) statistics.

Writes to stdout the long-form CSV with one row per (tier, payload):
    tier,payload_bytes,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns,ci_halfwidth

Also (when run with output dir as second arg) writes per-tier wide CSVs
suitable for the figure templates:
    <out>/tier_latency_smem.csv  (cols: payload_bytes, mean_ns, ci_halfwidth)
    <out>/tier_latency_dsmem.csv
    <out>/tier_latency_hbm.csv

These per-tier CSVs are what figures/tier-latency.tex consumes.
"""
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path


PAT = re.compile(
    r"PAPER_TIER_LATENCY\s+tier=(\w+)\s+payload=(\d+)\s+trial=\d+\s+ns=(\d+)"
)
TIER_ORDER = {"smem": 0, "dsmem": 1, "hbm": 2}


def percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    k = max(0, min(len(sorted_xs) - 1, int(round((p / 100.0) * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def main():
    if len(sys.argv) < 2:
        print("usage: extract.py <log_path> [<wide_out_dir>]", file=sys.stderr)
        sys.exit(1)
    log = sys.argv[1]
    wide_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    buckets = defaultdict(list)
    with open(log) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                tier = m.group(1)
                payload = int(m.group(2))
                ns = int(m.group(3))
                buckets[(tier, payload)].append(ns)

    rows = []
    for (tier, payload), xs in sorted(
        buckets.items(), key=lambda kv: (TIER_ORDER.get(kv[0][0], 99), kv[0][1])
    ):
        n = len(xs)
        if n == 0:
            continue
        mean = sum(xs) / n
        sd = sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
        sxs = sorted(xs)
        p50 = percentile(sxs, 50)
        p95 = percentile(sxs, 95)
        p99 = percentile(sxs, 99)
        ci = 1.96 * (sd / sqrt(n)) if n > 1 else 0.0
        rows.append((tier, payload, n, mean, sd, p50, p95, p99, ci))

    # Long-form CSV to stdout
    print("tier,payload_bytes,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns,ci_halfwidth")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{r[5]},{r[6]},{r[7]},{r[8]:.2f}")

    # Per-tier wide CSVs for the figure
    if wide_dir is not None:
        wide_dir.mkdir(parents=True, exist_ok=True)
        for tier in ("smem", "dsmem", "hbm"):
            with open(wide_dir / f"tier_latency_{tier}.csv", "w") as out:
                out.write("payload_bytes,mean_ns,ci_halfwidth\n")
                for r in rows:
                    if r[0] == tier:
                        out.write(f"{r[1]},{r[3]:.1f},{r[8]:.2f}\n")


if __name__ == "__main__":
    main()
