# Fan-lex-ai_Fairness_Utility_GraphLearning
Fairness–Utility Trade-offs in Graph Learning (Reformatted for Imperial/DSI)
[README.md](https://github.com/user-attachments/files/22451721/README.md)
# Fair Graph ML Starter (Node Classification & Link Prediction)

This is a **minimal, runnable** starter repo to showcase **Graph ML** with **basic fairness metrics** on Planetoid datasets (Cora/Citeseer/Pubmed) using **PyTorch Geometric**.

> Purpose: quickly produce **reproducible, inspectable results** (accuracy + simple fairness metrics) you can reference in your LIACS/CNS PhD application (Vacancy 15718).

## What’s inside
- **Node classification** (GCN) and **link prediction** (GCN encoder + dot-product decoder).
- A **synthetic sensitive attribute `S`** per node (for demo): `S=1` if node degree ≥ median, else `S=0`.
- **Fairness metrics** (simple): demographic parity (DP) gap and equalized odds (EO) gap.
- An optional **fairness penalty** (correlation between predictions and `S`) to illustrate a debiasing knob.
- Results saved under `results/` as CSV/JSON (via JSON only in this starter; you can add CSV later).

> ⚠️ Note: Public Planetoid datasets do **not** include real sensitive attributes; this repo demonstrates *workflow and metrics plumbing*. In a real project, replace `S` with a genuine attribute or a domain-justified proxy.

## Quick start

### 0) Create venv and install deps
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If `torch_geometric` wheel fails, follow its official install instructions for your platform/CUDA, then re-run `pip install -r requirements.txt` to add the rest.

### 1) Run node classification (Cora)
```bash
python src/train_nodeclf.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.0
python src/train_nodeclf.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.1
```

### 2) Run link prediction (Cora)
```bash
python src/train_linkpred.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.0
python src/train_linkpred.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.1
```

### 3) Inspect results
JSON files appear in `results/` with timestamps. Each run logs: accuracy/AUC and DP/EO gaps.

## Files
- `src/data.py` — dataset loading + synthetic sensitive attribute `S`
- `src/models.py` — GCN encoder and classifier
- `src/fairness.py` — DP/EO metrics + correlation penalty
- `src/utils.py` — seeds, logging
- `src/train_nodeclf.py` — training & evaluation for node classification
- `src/train_linkpred.py` — training & evaluation for link prediction

## Suggested figures for your email/application
- **Accuracy vs. fairness** trade-off curve by sweeping `--fair_lambda`.
- **DP/EO bars** before vs. after penalty.

## License
MIT
