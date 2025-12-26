# Decision Memo — Fairness–Utility Trade-offs in Graph Learning

**Repo:** https://github.com/LexAI-Fan/Fan-lex-ai_Fairness_Utility_GraphLearning  
**Release:** v1.0.0 — https://github.com/LexAI-Fan/Fan-lex-ai_Fairness_Utility_GraphLearning/releases/tag/v1.0.0  
**Last updated:** 2025-12-26

---

## 1) Problem & setting (LocalSBM + Planetoid)
This project studies a practical question: when adding a fairness objective to graph learning, how much utility is sacrificed, and where does the “knee” of the trade-off appear?  
Experiments run on a synthetic **LocalSBM** graph by default (fast, controlled), and the code path also supports **Planetoid** datasets (Cora/Citeseer/Pubmed) for a more realistic check.

## 2) Methods (nodeclf/linkpred + DP/EO + fair_lambda sweep)
Two tasks are covered: **node classification** and **link prediction**, each trained with a fairness regularizer controlled by a single knob `--fair_lambda (λ)`.  
Fairness is evaluated with **DP (Demographic Parity)** and **EO (Equalized Odds)** to keep the audit surface small while still capturing “selection parity” and “error parity” perspectives.

## 3) Reproducibility guarantees (one run → one JSON; scripted sweep)
Each run writes **one JSON artifact** containing configuration + metrics (the unit of evidence is a file, not a screenshot).  
A sweep is scripted via `scripts/run_sweep.ps1`, and plots are regenerated deterministically from saved artifacts with `python .\make_figures.py`.

## 4) Key findings (from docs/results_summary.md)
Across the tested λ values, **utility remains relatively stable up to λ≈0.3, then drops faster**—suggesting a knee region where fairness pressure becomes costly.  
In contrast, **DP improves early and then largely plateaus after λ≈0.3**, indicating diminishing fairness returns beyond that point.  
This pattern is visible in both node and link trade-off plots, with task-specific sensitivity.
See: docs/results_summary.md for the run table and plots.

## 5) Limitations (what this does *not* claim yet)
Planetoid runs require dataset download; if network access is restricted, the “real data” sweep may fail unless cached locally.  
DP/EO are intentionally simplified; they do not cover intersectional groups, richer sensitive attributes, or robustness under dataset shift.  
Results here are meant as a reproducible scaffold and a transparent baseline, not as a final fairness benchmark.

## 6) Next steps (what I would do to strengthen this as research)
Compute and plot the **Pareto frontier** from saved JSON artifacts (explicit multi-objective selection rather than eyeballing curves).  
Add stronger fairness suites (e.g., calibration-by-group, equal opportunity variants, intersectional slices) and more realistic sensitive-attribute definitions.  
Make Planetoid fully offline-friendly (download once, cache in `data/`, document a no-network workflow).

---

### Quick reproduce (Windows PowerShell)
```powershell
python .\src\train_nodeclf.py --dataset LocalSBM --epochs 200 --fair_lambda 0.3
python .\src\train_linkpred.py --dataset LocalSBM --epochs 50  --fair_lambda 0.3
python .\make_figures.py
# full sweep:
.\scripts\run_sweep.ps1
