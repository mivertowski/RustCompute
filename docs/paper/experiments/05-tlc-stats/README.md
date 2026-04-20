# Experiment 5 — TLC State-Space Stats

**Purpose**: validate the bounded-model-checking claims in §4.4, §6.7,
§7.6, §8.5, §9.2. Produces a table of (spec, parameter bound, distinct
states explored, wall-clock, invariants checked) — provides empirical
backing for the "mechanized in TLA+" claim.

**Hardware**: CPU only. Can run in parallel with GPU experiments.

**Wall-clock**: ~20 min (5 specs × ~4 min each at the configured bounds).

## Run

```bash
./run.sh <output_dir>      # called by run_all.sh
# or standalone:
./run.sh ./local_out
```

Internally:
1. Calls `../../verification/tlc.sh` for each spec.
2. Captures the TLC summary lines (states explored, distinct states,
   queue length, wall-clock).
3. Writes `tlc_stats.csv` with one row per spec.

## Output schema (`tlc_stats.csv`)

```
spec,bound_constants,distinct_states,states_generated,wall_clock_s,invariants_checked,result
actor_lifecycle,"|A|<=5,MaxSteps<=30",10300042,98472311,182.4,3,OK
hlc,"|N|<=3,MaxPhys<=5,MaxEvents<=10",184722,1248317,42.1,4,OK
k2k_delivery,"|A|<=3,MaxMsgs<=4,QueueCap<=2",55301,238417,8.7,5,OK
tenant_isolation,"|T|<=2,|K|<=3,MaxMsgs<=4",18472,72103,3.4,4,OK
migration,"MaxMsgs<=4",4823,18412,2.1,4,OK
multi_gpu_k2k,"|G|<=3,|A|<=4,MaxMsgs<=4",73104,328719,11.5,2,OK
```

## Failure modes

If a spec runs out of memory, drop the constant in the corresponding
`.cfg` by 1 (e.g., `MaxMsgs = 4 → 3`). Re-run; record the new bounds in
the CSV's `bound_constants` column.

## Feeds

- `docs/paper/figures/data/tlc_stats.csv`
- `tab:tlc-stats` table (to be added to the paper's evaluation section
  or appendix; template in `docs/paper/figures/tlc-stats.tex`).
