# Decision memo — Fairness–Utility Trade-offs in Graph Learning

**Owner:** Fan Xuejiao (Ariel)  
**Repo:** https://github.com/LexAI-Fan/Fan-lex-ai_Fairness_Utility_GraphLearning
**Release:** https://github.com/LexAI-Fan/Fan-lex-ai_Fairness_Utility_GraphLearning/releases/tag/v1.0.0 
**Last updated:** 2025-12-25

## 1) Problem & setting
This mini-study asks a practical question: when we regularise for group fairness in graph learning, what utility do we give up, and what does the trade-off curve look like across tasks?  
I focus on a controlled synthetic graph (**LocalSBM**) to make behaviour easy to interpret, and keep support for Planetoid datasets (**Cora/Citeseer/Pubmed**) for external validity once the environment can reliably download/cache them.

## 2) Methods
The repository provides two training pipelines: `train_nodeclf.py` (node classification; utility = accuracy) and `train_linkpred.py` (link prediction; utility = AUC/AP).  
Fairness is evaluated with two lightweight group metrics: **DP (Demographic Parity)** and **EO (Equalized Odds)**. A single regularisation knob `--fair_lambda` is swept over a small grid to trace the utility–fairness frontier, rather than selecting one “best” setting.

## 3) Reproducibility guarantees
Each run writes **one JSON artifact** containing configuration + metrics to `results/`, so every number in the figures can be traced back to a concrete run.  
The sweep is scripted (`scripts/run_sweep.ps1`) to minimise manual command drift. Figures are regenerated from JSON only (`make_figures.py`), which keeps plotting deterministic and audit-friendly.

## 4) Key findings (summary)
The fairness–utility relationship is not linear. In the current sweep, **utility changes are small up to λ ≈ 0.3, then drop faster** as regularisation increases.  
On the fairness side, **DP improves early and then plateaus after roughly λ ≈ 0.3**, suggesting diminishing returns beyond that point.  
Across tasks, the same qualitative pattern appears: the “interesting region” is around moderate λ values where fairness moves meaningfully but utility remains relatively stable.

## 5) Limitations
Planetoid runs can fail in restricted networks because dataset download/caching depends on external hosts; this is an environment constraint rather than a modelling issue.  
DP/EO are intentionally simplified summaries of group behaviour, and the “sensitive attribute” in LocalSBM is synthetic. The goal here is a clean, reproducible scaffold rather than a full socio-technical fairness assessment.

## 6) Next steps
1) **Pareto frontier:** compute non-dominated points from the JSON artifacts (max utility, min unfairness) and annotate them in the plots for clearer model selection.  
2) **Real datasets (robustly):** add an offline-friendly path (document cache location + optional manual download) and rerun the same sweep on Cora/Citeseer/Pubmed.  
3) **Richer evaluation:** extend beyond DP/EO (e.g., additional fairness metrics and basic uncertainty/calibration checks) and run a small sensitivity check so the trade-off story is not tied to one setting.

