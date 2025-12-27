# Pilot HCI Study: Static Summary vs Interactive Trade-off Explorer (N=6)

**Goal.** Evaluate whether an interactive trade-off explorer helps people make constrained fairness–utility decisions more efficiently and accurately than a static results summary (table).

## Participants
N=6 (anonymous, Dec 2025). Self-reported ML/data familiarity:
- Almost none: 2
- Some familiarity: 2
- Frequent use: 1
- Professional/research: 1

## Design and Tasks
Within-subject, counterbalanced:
- Version A: static-first → interactive
- Version B: interactive-first → static

Two tasks per participant: select the best model point under explicit constraints on fairness (demographic parity gap) and utility (accuracy).

## Measures
- **Decision time** (seconds; self-reported)
- **Error rate** (1 if constraint violated or non-optimal under constraint; else 0)
- **Subjective ratings** (1–7 Likert): confidence, mental effort, ease of decision

## Results

### Objective performance

| Condition | Avg. time per task (s) | Error rate (2 tasks × N=6) |
|---|---:|---:|
| Static summary (table) | 46.3 | 33% (4/12 incorrect) |
| Interactive explorer | 24.8 | 8% (1/12 incorrect) |

Participants were faster with the interactive explorer (46.3s → 24.8s; ~46% reduction) and made fewer incorrect choices (33% → 8%).

### Subjective ratings (median, 1–7)

| Item | Static | Interactive |
|---|---:|---:|
| Confidence in choices | 5 | 6.5 |
| Mentally taxing | 5 | 3 |
| Made trade-off decisions easier | 4 | 7 |

### Qualitative feedback (selected)
- “The interactive explorer lets you see the trade-off instantly—much clearer than scanning rows in a table.”
- “Static table is okay for a quick overview, but finding the exact point under constraints took a lot of back-and-forth.”
- “Hovering and filtering in the UI made it easy to confirm constraints were met.”
- “Really liked the real-time update; felt more in control of the decision.”

## Interpretation
In this small pilot, the interactive trade-off explorer improved both efficiency and accuracy while reducing perceived mental effort compared to a static results summary. This supports the use of interactive decision support for governance-style model selection where constraints matter.

## Limitations
- Pilot sample (N=6), self-reported timing, and simplified fairness constraints/metrics.
- Results should be treated as directional evidence; future work should collect larger samples and log timings automatically.

## Next steps
- Log timing and interactions automatically in the UI (no self-report).
- Add Pareto frontier mode and additional fairness metrics (EO, intersectional metrics).
- Test more realistic “sensitive attribute” definitions and reporting templates for audit contexts.
