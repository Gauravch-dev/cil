# Current release review

The public interface has exactly two pages, Rainfall and Flood. Rainfall shows one source-backed chart at a time; Flood shows a manual form beside the assessment and river chart. Model details and evaluation metrics stay in documentation. Previously developed analysis interfaces and experiments remain preserved outside public navigation.

The overall requested 75% accuracy/precision/recall/F1 target is met on the retrospective test. It is **not** met for new warning crossings: one positive example is insufficient, precision is 33.33% and F1 50%. Simple physical baselines outperform ML overall. This is a transparent academic prototype, not a validated operational warning system. Complete rainfall and more independent crossing observations remain the data priorities. See MODEL_REPORT.md for exact definitions, counts and limitations.
