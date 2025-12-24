# Fairness–Utility Trade-offs in Graph Learning

This repository provides a small, reproducible scaffold for studying utility–fairness trade-offs in graph learning.

Current default runs use a synthetic graph setting (LocalSBM) and log metrics to JSON artifacts under results/.

It includes:
- node classification and link prediction experiments
- simple fairness metrics (DP, EO)
- a controllable regularization knob (lambda / fair_lambda)
- result artifacts saved as JSON under results/


The focus is reproducibility and auditability: one command produces one JSON artifact with configuration and metrics.

## Contents

- src/
  - data.py: data loading / graph generation
  - models.py: GNN models
  - fairness.py: DP/EO metrics and (optional) penalty term
  - utils.py: seeds, logging, helpers
  - train_nodeclf.py: node classification training and evaluation
  - train_linkpred.py: link prediction training and evaluation
- results/
  - experiment outputs (JSON)
- make_figures.py
  - figure generation from saved JSON artifacts
- docs/
  - figures and a small example artifact for review

## Environment setup (Windows PowerShell)

From the repository root:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r .\requirements.txt

If PowerShell blocks script activation, run once:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

## Run experiments

First, inspect available arguments:

python .\src\train_nodeclf.py -h
python .\src\train_linkpred.py -h

Example (node classification):

python .\src\train_nodeclf.py --epochs 50

Example (link prediction):

python .\src\train_linkpred.py --epochs 50

All runs write a JSON artifact into results/.

## Evidence (one reproducible run)

Command used (Windows PowerShell):

python .\src\train_nodeclf.py --epochs 50

Example output artifact:

docs\example_run_seed0.json

Key fields recorded in the JSON:

- acc (or auc): <fill from JSON>
- dp_gap: <fill from JSON>
- eo_gap: <fill from JSON>

## Evidence (one reproducible run)

Windows PowerShell:

python .\src\train_nodeclf.py --epochs 200

Example output artifact:

docs\example_run_seed0.json
(original run file: results\nodeclf_LocalSBM_lambda0.0_degree_20251224-151142.json)

Key fields:

- test_acc: 1.0000
- dp: 0.0487
- eo: 0.0000


## Making figures

If you have multiple JSON artifacts in results/, you can generate plots via:

python .\make_figures.py

Figures will be written to docs/figures/ (or the output path defined in the script).

## Notes and limitations

- If a synthetic group attribute is used, it is for demonstrating the evaluation pipeline and should not be interpreted as a protected attribute.
- DP/EO gaps here are simple baselines intended for transparency and quick inspection.
- Graph fairness is sensitive to graph structure and data assumptions; interpret results with the stated configuration in each JSON artifact.

## License

MIT
