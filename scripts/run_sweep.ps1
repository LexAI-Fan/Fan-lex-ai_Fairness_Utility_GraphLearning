# scripts/run_sweep.ps1
# Reproducible sweep for LocalSBM (and optional real datasets)
$ErrorActionPreference = "Stop"

# knobs
$L = @(0.0, 0.1, 0.3, 1.0)
$epochsNode = 200
$epochsLink = 50

# default: LocalSBM only (offline-friendly)
$datasets = @("LocalSBM")

foreach ($d in $datasets) {
  foreach ($x in $L) {
    python .\src\train_nodeclf.py --dataset $d --epochs $epochsNode --fair_lambda $x
    python .\src\train_linkpred.py --dataset $d --epochs $epochsLink --fair_lambda $x
  }
}

python .\make_figures.py
Write-Host "Done. Figures saved to docs/figures/ (or repo root if script outputs there)."
