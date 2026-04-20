#!/usr/bin/env python3
"""Parse paper_snapshot_restart stdout into CSV.

Stdout: long-form (size, phase, n, mean, sd, p50, p95, p99).
Optional second arg <out_dir>: writes snapshot_restart_wide.csv with cols
(size, capture_us, copy_us, ack_us) suitable for figures/snapshot-restart.tex.
"""
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

PAT = re.compile(
    r"PAPER_SNAPSHOT_RESTART\s+size=(\d+)\s+trial=\d+\s+phase=(\w+)\s+ns=(\d+)"
)


def percentile(xs, p):
    if not xs:
        return 0
    return sorted(xs)[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


def humansize(n):
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n}{unit}" if unit != "B" else str(n)
        n //= 1024
    return f"{n}GiB"


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
                buckets[(int(m.group(1)), m.group(2).strip())].append(int(m.group(3)))

    print("state_size_bytes,phase,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns")
    sizes_seen = set()
    means_by_size_phase = defaultdict(dict)
    for (size, phase), xs in sorted(buckets.items()):
        n = len(xs); mean = sum(xs) / n
        sd = sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
        print(f"{size},{phase},{n},{mean:.1f},{sd:.1f},"
              f"{percentile(xs,50)},{percentile(xs,95)},{percentile(xs,99)}")
        sizes_seen.add(size)
        means_by_size_phase[size][phase] = mean / 1000.0  # ns -> us

    if wide_dir is not None:
        wide_dir.mkdir(parents=True, exist_ok=True)
        with open(wide_dir / "snapshot_restart_wide.csv", "w") as out:
            out.write("size,capture_us,copy_us,ack_us\n")
            for size in sorted(sizes_seen):
                m = means_by_size_phase[size]
                out.write(f"{humansize(size)},{m.get('capture',0):.1f},"
                          f"{m.get('copy',0):.1f},{m.get('ack',0):.1f}\n")


if __name__ == "__main__":
    main()
