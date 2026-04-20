#!/usr/bin/env python3
"""Parse the existing sustained_throughput integration-test stdout
into per-trial per-window CSV.

Stdout: long-form (trial, window_secs, throughput_Mops_s, p50, p95, p99, ops).
Optional second arg <out_dir>: writes sustained_t<N>.csv per trial,
suitable for figures/sustained-cv.tex.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

# Match lines like:  "      5s      10.42M       70       80      100   52107391"
PAT = re.compile(
    r"^\s*([0-9.]+)s\s+([0-9.]+)M\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: extract.py <trial1.log> [trial2.log ...] [--wide=<dir>]")

    args = sys.argv[1:]
    wide_dir = None
    logs = []
    for a in args:
        if a.startswith("--wide="):
            wide_dir = Path(a.split("=", 1)[1])
        else:
            logs.append(a)

    print("trial,window_secs,throughput_Mops_s,p50_ns,p95_ns,p99_ns,ops")
    by_trial = defaultdict(list)
    for path in logs:
        try:
            trial = int(re.search(r"trial(\d+)", Path(path).name).group(1))
        except Exception:
            trial = 0
        with open(path) as f:
            for line in f:
                m = PAT.match(line)
                if m:
                    secs, mops, p50, p95, p99, ops = m.groups()
                    print(f"{trial},{secs},{mops},{p50},{p95},{p99},{ops}")
                    by_trial[trial].append((secs, mops, p50, p95, p99, ops))

    if wide_dir is not None:
        wide_dir.mkdir(parents=True, exist_ok=True)
        for trial, rows in by_trial.items():
            with open(wide_dir / f"sustained_t{trial}.csv", "w") as out:
                out.write("window_secs,throughput_Mops_s,p50_ns,p95_ns,p99_ns,ops\n")
                for r in rows:
                    out.write(",".join(r) + "\n")


if __name__ == "__main__":
    main()
