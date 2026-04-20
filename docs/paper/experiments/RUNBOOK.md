# Paper Experiments Runbook

Six experiments that strengthen the empirical claims of *Persistent GPU
Actors*. Each generates a CSV that feeds a pgfplots figure in the paper,
plus a `notes.md` capturing trial conditions.

## Pre-flight (do this once on VM spin-up)

```bash
# 1. Verify hardware
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
# Expected: 2 × NVIDIA H100, CC 9.0, 80+ GB each.

# 2. NVLink topology
nvidia-smi nvlink --status
# Expected: GPU0 ↔ GPU1 over multiple NVLink lanes.

# 3. Lock clocks + exclusive mode (REQUIRES sudo)
sudo nvidia-smi -pm 1                                # persistence mode
sudo nvidia-smi -c EXCLUSIVE_PROCESS                 # exclusive compute
sudo nvidia-smi -lgc 1785                            # H100 boost clock
# (If 1785 MHz refused on your SKU, query: nvidia-smi -q -d SUPPORTED_CLOCKS)

# 4. CPU governor
sudo cpupower frequency-set -g performance || true   # no-op if not installed

# 5. Build everything once (cached for all six experiments)
cd /path/to/RustCompute
cargo build --workspace --features cuda --release \
    --exclude ringkernel-txmon
cargo build -p ringkernel-cuda --features "cuda,cooperative,multi-gpu" --release
```

Expected total pre-flight wall-clock: 10–15 min (mostly the cargo build).

## Experiment matrix

| # | Experiment              | Wall-clock | GPUs | Paper §          | Output figure |
|---|-------------------------|------------|------|------------------|---------------|
| 1 | K2K tier latency        | ~25 min    | 1    | §6.1, §10        | `fig:tier-latency`     |
| 2 | Snapshot/restart        | ~15 min    | 1    | §10.5            | `fig:snapshot-restart` |
| 3 | Lifecycle overhead      | ~15 min    | 1    | §4               | `fig:lifecycle-cost`   |
| 4 | Sustained timeseries    | ~70 min*   | 1    | §10.2            | `fig:sustained-cv`     |
| 5 | TLC state-space stats   | ~20 min    | 0    | §4.4, §6.7, etc. | `tab:tlc-stats`        |
| 6 | NVLink P2P migration    | ~30 min    | 2    | §9.3             | `fig:migration-cost`   |

*Experiment 4 includes a 60-second sustained run (already established
methodology); the 70-min budget covers warmup + 4 trials.

Total budget: ~3 hours of GPU time + 20 min for pre-flight. Headroom for
1 hour of debugging/re-runs.

## Order of operations (recommended)

Run in this order so a fault in one does not invalidate later ones:

1. **Pre-flight** (`nvidia-smi`, build, clock lock).
2. **Exp 5 — TLC stats** (CPU only; can run while pre-flight is in
   progress; saves GPU time). See `05-tlc-stats/`.
3. **Exp 1 — Tier latency**. See `01-tier-latency/`.
4. **Exp 2 — Snapshot/restart**. See `02-snapshot-restart/`.
5. **Exp 3 — Lifecycle overhead**. See `03-lifecycle-overhead/`.
6. **Exp 4 — Sustained timeseries**. See `04-sustained-timeseries/`.
   (Long run; let it finish unattended.)
7. **Exp 6 — NVLink P2P migration**. See `06-nvlink-migration/`.
   (Requires both GPUs idle.)
8. **Archive + commit**: `./run_all.sh --archive-only`.

Each experiment directory has its own `README.md` with the exact `cargo`
or shell command, the expected output schema, and the figure-template
that consumes the resulting CSV.

## Quick path: one-shot runner

If everything is wired up correctly, the top-level `run_all.sh` does the
above end to end:

```bash
cd docs/paper/experiments
./run_all.sh
```

It writes results to `results/<timestamp>/` and prints a summary.
Re-runnable; results from earlier runs are preserved under their
timestamp directory.

## After the run

1. Each experiment's `extract.py` converts Criterion JSON → CSV.
   `run_all.sh` invokes them automatically; for ad-hoc reruns:
   ```bash
   cd docs/paper/experiments/01-tier-latency
   python3 extract.py ../../../target/criterion/paper_k2k_tier_latency \
       > results/tier-latency.csv
   ```
2. Figure templates in `figures/` consume those CSVs:
   ```bash
   cd docs/paper/figures
   pdflatex tier-latency.tex     # produces tier-latency.pdf
   ```
3. Drop the resulting PDFs into `docs/paper/figures/` and reference
   them from the relevant section. The paper's main `Makefile` will
   pick up the new `.pdf` figures automatically on next `make`.

## What to capture besides numbers

For every run, record:
- Full `nvidia-smi -q` snapshot at start (saved as
  `results/<timestamp>/nvidia_smi_full.txt` by `run_all.sh`).
- Git commit (`git rev-parse HEAD`).
- Driver version, CUDA toolkit version, Rust version.
- ECC error counts before and after (any non-zero invalidates the trial).
- Wall-clock for the experiment itself.

`run_all.sh` collects all of these into a `manifest.json`; the paper's
reproducibility section references this artifact directly.

## Failure-mode triage

| Symptom                                | Likely cause                 | Action                                 |
|----------------------------------------|------------------------------|----------------------------------------|
| "PTX module load failed"               | sm_90 PTX missing            | `cargo clean -p ringkernel-cuda; rebuild --features cooperative` |
| "P2P access not enabled"               | NVLink not initialized       | `enable_peer_access(0,1)` already in bench; check nvlink topology with `nvidia-smi nvlink --status` |
| "EXCLUSIVE_PROCESS holds device"       | another process on the GPU   | `sudo nvidia-smi -c DEFAULT` (run, then re-set EXCLUSIVE) |
| Criterion latencies vary >5% between trials | thermal throttle      | Re-check `nvidia-smi -q -d TEMPERATURE`; pause 60s between trials |
| TLC out-of-memory                      | bounded model too big        | Drop the `<spec>.cfg` constants by 1 (e.g., `MaxMsgs = 4 → 3`) |

## Quick links

- Paper source: `../`
- Existing benchmark methodology doc: `../../benchmarks/METHODOLOGY.md`
- Existing H100 baseline: `../../benchmarks/h100-b200-baseline.md`
- TLA+ specs: `../../verification/`
