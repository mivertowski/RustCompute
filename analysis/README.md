# RingKernel analysis scripts

Local-only tooling for paper-prep analysis. Not part of any crate build.

## b200_h100_compare.py

Reads Criterion `estimates.json` outputs from H100 baseline and B200
results, emits a markdown comparison table suitable for paper Addendum 7.

Usage:

```bash
python3 analysis/b200_h100_compare.py \
    --h100 benchmark_results/h100/criterion \
    --b200 benchmark_results/b200/<bench>/<ts>/criterion \
    --out  docs/paper/sections/_b200_h100_table.md
```

If matplotlib is installed, also emits per-bench bar plots to the
parent dir of `--out`. Plotting is best-effort and silently skipped
if matplotlib is unavailable — the markdown output is the primary
deliverable.
