# Fairness–Utility Trade-offs in Graph Learning

This repository provides a small, reproducible scaffold for studying **utility–fairness trade-offs** in graph learning.

Current default runs use a **synthetic graph setting (LocalSBM)** and log metrics to **JSON artifacts under `results/`**.

It includes:
- Node classification and link prediction experiments
- Simple fairness metrics (**DP**, **EO**) for auditing
- A controllable fairness regularization knob (**`fair_lambda`**)
- Per-run JSON artifacts saved under `results/` (config + metrics)

The focus is **reproducibility and auditability**: **one command produces one JSON artifact** containing configuration + metrics.

---

## Description (3 lines)

- **What:** Runs reproducible LocalSBM graph experiments for node classification and link prediction with fairness regularization (`fair_lambda`), logging utility + fairness metrics (Acc/AUC, DP/EO).
- **How:** `python .\src\train_nodeclf.py --epochs 50` and `python .\src\train_linkpred.py --epochs 50`; generate plots via `python .\make_figures.py`.
- **Outputs:** JSON artifacts in `results/`, example artifacts in `docs/example_*.json`, and figures in `docs/figures/fig_node_tradeoff.png` + `docs/figures/fig_link_tradeoff.png`.
**

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
