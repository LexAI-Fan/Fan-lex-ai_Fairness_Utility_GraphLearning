# Fairness–Utility Trade-offs in Graph Learning

This repository provides a small, reproducible scaffold for studying **utility–fairness trade-offs** in graph learning.

Default experiments use a **synthetic LocalSBM graph** and log **configuration + metrics** as **one JSON artifact per run** under `results/`.

## Description 

What: Reproducible LocalSBM graph experiments (node classification + link prediction) with a fairness regularizer (fair_lambda) and DP/EO evaluation.

How: python .\src\train_nodeclf.py --epochs 50 and python .\src\train_linkpred.py --epochs 50; then python .\make_figures.py.

Outputs: Per-run JSON artifacts in results/, example artifacts in docs/example_*.json, and trade-off plots in docs/figures/.

---

## What’s included

- Node classification and link prediction experiments
- Simple fairness metrics: **DP** (Demographic Parity), **EO** (Equalized Odds)
- A controllable fairness regularization knob: `--fair_lambda`
- Reproducible run artifacts: JSON logs saved under `results/`

The focus is **reproducibility and auditability**: *one command → one JSON artifact* (config + metrics).

## Repository layout

- `src/`
  - `data.py`: data loading / synthetic graph generation
  - `models.py`: GNN models
  - `fairness.py`: DP/EO metrics and (optional) penalty term
  - `utils.py`: seeds, logging, helpers
  - `train_nodeclf.py`: node classification training and evaluation
  - `train_linkpred.py`: link prediction training and evaluation
- `results/`
  - experiment outputs (JSON artifacts; typically not committed)
- `make_figures.py`
  - figure generation from saved JSON artifacts
- `docs/`
  - `figures/` and small example artifacts for review

---

## Environment setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r .\requirements.txt
