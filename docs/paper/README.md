# Persistent GPU Actors — Paper Source

> **Persistent GPU Actors: A Formal Model for Living Computation on the Device**
> Michael Ivertowski, *Independent Researcher* (Zurich)
> Target venue: arXiv pre-print, then SSRN; submission to a programming-languages or systems venue thereafter.

This directory contains the LaTeX source of the RingKernel companion paper.
The paper formalizes the persistent GPU actor model that RingKernel
implements, mechanizes the safety properties in TLA+, and validates the
formal claims against H100 NVL silicon.

## Structure

```
paper/
├── main.tex                     # Document root, preamble, custom commands
├── references.bib               # Bibliography
├── Makefile                     # Build automation
├── README.md                    # This file
├── .gitignore                   # LaTeX build artifacts
├── figures/                     # PDF/TikZ figures (currently empty)
└── sections/
    ├── abstract.tex             # Abstract: 5 contributions, headline numbers
    ├── introduction.tex         # The Gap, What Hopper Made Possible, Contribution
    ├── background.tex           # Actor model, GPU conventions, persistent kernels
    ├── related_work.tex
    ├── model.tex                # Lifecycle SOS + Lifecycle-Soundness theorem
    ├── memory.tex               # Memory hierarchy, axiomatic constraints
    ├── k2k.tex                  # K2K calculus + delivery theorem
    ├── supervision.tex          # HLC + restart-monotonicity theorem
    ├── tenancy.tex              # Audit boundaries + cross-tenant isolation
    ├── implementation.tex       # RingKernel runtime, CUDA driver surface
    ├── evaluation.tex           # H100 NVL benchmarks
    ├── discussion.tex           # Limits, future work
    ├── conclusion.tex
    └── appendix.tex             # Quiesce/Terminate/Fail SOS rules
```

## Build

```bash
make            # full build with bibliography (4 LaTeX passes)
make quick      # single LaTeX pass for fast iteration
make watch      # latexmk -pvc continuous build
make clean      # remove auxiliary files (keeps PDF)
make distclean  # remove everything
```

Required tools: `pdflatex`, `bibtex`, optionally `latexmk` for `watch`.
On Debian/Ubuntu: `apt install texlive-latex-recommended texlive-latex-extra
texlive-fonts-recommended texlive-bibtex-extra latexmk`.

## Companion artifacts

The paper's formal claims are mechanically backed by the TLA+ specifications
in `../verification/`:

| Theorem (paper)                  | TLA+ spec (`../verification/`) |
|----------------------------------|--------------------------------|
| Lifecycle Soundness (Thm 4.1)    | `actor_lifecycle.tla`          |
| K2K Delivery Properties          | `k2k_delivery.tla`             |
| HLC Restart Monotonicity         | `hlc.tla`                      |
| Cross-Tenant Isolation           | `tenant_isolation.tla`         |
| Migration Safety (deferred)      | `migration.tla`                |
| Multi-GPU K2K (deferred)         | `multi_gpu_k2k.tla`            |

Run `make verify` (or `./tlc.sh` from `../verification/`) to model-check
all specs.

The empirical numbers in the Evaluation section are taken from
`../benchmarks/ACADEMIC_PROOF.md` (run on H100 NVL, 2026-04-16); raw
Criterion data lives under `target/criterion/` in the repo root.

## Style

The preamble mirrors the DataSynth paper at
`/home/michael/DEV/Repos/RustSyntheticData/SyntheticData/paper/main.tex`
for visual consistency: 11pt a4paper, lmodern, microtype, semantic SOS
rules, theorem environments via `amsthm`, hyperref + cleveref for
cross-references, natbib (numbers, sort&compress) for citations.

The `\system{}` macro renders the system name (currently
`\textsc{RingKernel}`); change this in `main.tex` to retarget the paper
without editing every section.

## Status

Drafting in progress. Sections currently full:

- `abstract.tex`, `introduction.tex`, `model.tex` — complete first draft.
- `background.tex` — first subsection complete; remaining are stubs.
- All other sections are stubs that outline content + pointers to source
  material (TLA+ specs, benchmark data, runtime modules).
