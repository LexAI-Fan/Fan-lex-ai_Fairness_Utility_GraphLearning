\# Results Summary (LocalSBM)



\## Setup

\- Tasks: node classification, link prediction

\- Dataset: LocalSBM (synthetic)

\- Sweep: fair\_lambda ∈ {0.0, 0.1, 0.3, 1.0}

\- Artifacts: one JSON per run in `results/`



\## Key observations

1\) Utility vs fairness: (写一句你看到的趋势，比如 fair\_lambda 增大后 dp/eo 是否下降、acc/auc 是否变化)

2\) Diminishing returns / knee point: (有没有某个 lambda 之后收益变小但代价变大)

3\) Task difference: nodeclf 与 linkpred 的 trade-off 是否一致



\## Files produced

\- Example artifacts: `docs/example\_nodeclf\_seed0.json`, `docs/example\_linkpred\_seed0.json`

\- Figures: `docs/figures/fig\_node\_tradeoff.png`, `docs/figures/fig\_link\_tradeoff.png`



