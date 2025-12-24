# Fairness–Utility Trade-offs in Graph Learning (Reproducible Mini-Study)

This repository implements a small, reproducible study of utility–fairness trade-offs for graph learning using PyTorch Geometric on Planetoid datasets (Cora/CiteSeer/PubMed).

It is intentionally minimal: the goal is to provide an auditable baseline with clear outputs that I can extend in PhD work.

## Research question

When training GNNs for node classification and link prediction, how does a simple fairness regularizer (a correlation penalty between predictions and a group attribute) shift the trade-off between:
- utility (accuracy or AUC), and
- fairness gaps (DP gap and EO gap)?

Because Planetoid does not include sensitive attributes, this repo uses a synthetic group attribute to demonstrate the full pipeline from training to fairness reporting.

## What I built

- Two tasks
  - Node classification with a GCN (metric: accuracy)
  - Link prediction with a GCN encoder + dot-product decoder (metric: AUC)
- Group attribute (demo)
  - S = 1 if node degree >= median, else S = 0
- Fairness metrics (simple, transparent baselines)
  - DP gap: difference in positive prediction rates across groups
  - EO gap: difference in error rates across groups
- Regularization knob
  - correlation penalty between predictions and S, controlled by fair_lambda
- Reproducible outputs
  - each run writes one timestamped JSON file under results/ containing config and metrics

## Why synthetic S is used

Planetoid datasets do not contain real sensitive attributes.
The synthetic S is used to verify the engineering pipeline (data -> model -> metrics -> artifact logging).
In real applications, S should be replaced by a genuine attribute or a domain-justified proxy, with an explicit threat model and ethics review.

## Quick start

### 0) Environment

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

If torch_geometric wheels fail, install PyG following its official instructions for your platform/CUDA, then re-run:
pip install -r requirements.txt

### 1) Node classification

python src/train_nodeclf.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.0 --seed 0
python src/train_nodeclf.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.1 --seed 0

### 2) Link prediction

python src/train_linkpred.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.0 --seed 0
python src/train_linkpred.py --dataset Cora --epochs 200 --hidden 64 --fair_lambda 0.1 --seed 0

### 3) Results

Run artifacts are saved under results/ as timestamped JSON files.
Each record includes:
- dataset, task, seed, hyperparameters, timestamp
- utility metric (accuracy or AUC)
- fairness gaps (DP gap, EO gap)

## Experimental protocol (recommended for reporting)

For any figure/table you include in applications:
- run at least 3 seeds (e.g., 0, 1, 2)
- sweep fair_lambda (e.g., 0.0, 0.01, 0.05, 0.1, 0.2)
- report mean and standard deviation of utility and fairness gaps
- keep dataset split settings and hidden size fixed unless explicitly studying them

A simple plot set:
- utility vs DP gap across fair_lambda
- utility vs EO gap across fair_lambda
- bar chart of DP/EO before vs after regularization (same seed and hyperparams)

## Repository layout

- src/data.py
  - dataset loading
  - synthetic group attribute S
- src/models.py
  - GCN encoder and task heads
- src/fairness.py
  - DP/EO gap metrics
  - correlation penalty
- src/utils.py
  - seeding and run logging
- src/train_nodeclf.py
  - train/eval for node classification
- src/train_linkpred.py
  - train/eval for link prediction

## Limitations

- The group attribute S is synthetic and does not represent a protected attribute.
- DP/EO are simple gap metrics; they are useful as baselines but not sufficient for a full fairness assessment.
- Graph fairness is sensitive to graph structure (homophily), label bias, and sampling choices; results here should be interpreted as pipeline validation and baseline behavior.

## Next steps (scoped)

Short-term improvements that fit this repo's current scope:

- Make results more stable:
  - run multiple seeds and report mean/std (export a small summary JSON or CSV)
- Make sweeps easier:
  - add a simple sweep script for fair_lambda and a plotting script for trade-off curves
- Make outputs easier to review:
  - document the JSON output keys and add a minimal example artifact under docs/

Longer-term directions (not implemented here yet):

- Evaluate on datasets with real group attributes (or a justified proxy) and document the threat model.
- Add additional baselines (e.g., GraphSAGE/GAT) once the evaluation harness is stable.

## Citation

If you want to reference this repository:
Fan Xuejiao. Fairness–Utility Trade-offs in Graph Learning (code repository). GitHub, 2025.

## Contact

Open an issue on GitHub for questions or reproduction requests.


