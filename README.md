# Fairness–Utility Trade-offs in Graph Learning

This repository provides a small, reproducible scaffold for studying **utility–fairness trade-offs** in graph learning.

Current default runs use a **synthetic graph setting (LocalSBM)** and log metrics to **JSON artifacts under `results/`**.

It includes:
- Node classification and link prediction experiments
- Simple fairness metrics (**DP**, **EO**)
- A controllable regularization knob (**`lambda` / `fair_lambda`**)
- Result artifacts saved as JSON under `results/`

The focus is **reproducibility and auditability**: **one command produces one JSON artifact** containing configuration + metrics.

---

## Contents

- `src/`
  - `data.py`: data loading / synthetic graph generation
  - `models.py`: GNN models
  - `fairness.py`: DP/EO metrics and (optional) penalty term
  - `utils.py`: seeds, logging, helpers
  - `train_nodeclf.py`: node classification training and evaluation
  - `train_linkpred.py`: link prediction training and evaluation
- `results/`
  - experiment outputs (JSON artifacts)
- `make_figures.py`
  - figure generation from saved JSON artifacts
- `docs/`
  - `figures/` and a small example artifact for review

---

## Environment setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r .\requirements.txt
