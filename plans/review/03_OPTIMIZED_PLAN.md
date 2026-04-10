# OPTIMIZED PLAN: Revised for Publication Readiness

**Reviewer Note:** This replaces the original article_plan.md and IMPLEMENTATION_PLAN.md.  
**All changes from the original plan are marked with `[CHANGED]` or `[NEW]`.**  
**Deleted/removed content from the original is marked with `[REMOVED]`.**  
**Pass this file to your implementation agent.**

---

## Title

`[CHANGED]` **Old:** "Benchmarking Time-Series Foundation Models for Industrial Predictive Maintenance: Critical Limitations Exposed"  
`[CHANGED]` **New:** "An Empirical Evaluation of Time-Series Foundation Models for Industrial Predictive Maintenance Tasks"

**Rationale:** The original title overclaims ("Critical Limitations Exposed" implies comprehensive evidence you don't have). The new title is accurate and professional.

---

## Scope Reduction

`[CHANGED]` The original plan was too ambitious for a single-author proceedings paper. Here is the revised scope:

### Models (4 instead of 9)

| Model | Type | Rationale for Inclusion |
|-------|------|------------------------|
| MOMENT | Foundation (385M) | Best multivariate TSFM, LoRA-compatible, already implemented |
| Chronos | Foundation (710M) | Amazon's flagship, tokenization-based, already implemented |
| Lag-Llama | Foundation (7M) | `[CHANGED]` Replaced TimeGPT — lightweight, open-source, probabilistic |
| PatchTST | Baseline | `[CHANGED]` Must be PROPERLY implemented (not stub) |

`[REMOVED]` Sundial — not implemented, adds marginal value  
`[REMOVED]` Time-MoE — not implemented, adds marginal value  
`[REMOVED]` TimeGPT — API-only model, non-reproducible, raises reviewer concerns about reproducibility  
`[REMOVED]` Autoformer — not implemented, not a TSFM  
`[REMOVED]` Vanilla Transformer — not implemented, not a TSFM  

### Datasets (3 instead of 6)

| Dataset | Domain | Tasks | Rationale |
|---------|--------|-------|-----------|
| C-MAPSS (FD001-FD004) | Turbofan | RUL + Forecasting | Gold standard PdM dataset, 4 subsets for cross-domain |
| MIMII | Factory Machinery | Anomaly Detection | Multi-machine type, cross-machine transfer |
| Wind SCADA | Wind Turbines | Forecasting | Real SCADA data, your domain expertise |

`[REMOVED]` PHM Milling — lower priority, adds page count without insight  
`[REMOVED]` PU Bearings — requires manual data request, unreliable availability  
`[REMOVED]` PRONOSTIA — too small (17 samples), results would be unreliable  

### Scenarios (2 instead of 3)

| Scenario | Details |
|----------|---------|
| Zero-shot | All 4 models × 3 datasets — direct inference |
| Few-shot (LoRA) | MOMENT + Lag-Llama × 3 datasets — 1% training data |

`[REMOVED]` Full cross-domain transfer matrix — too compute-intensive for proceedings paper. Instead, include ONE cross-domain experiment as a case study: C-MAPSS FD001 → FD002/FD003/FD004.

---

## Revised Paper Outline (6-8 pages, IEEE conference format)

### Abstract (150 words)
`[CHANGED]` Remove ALL specific numbers (35%, 62%, 18%) until experiments are actually run. Write the abstract LAST, after results are in hand.

### Section 1: Introduction (0.8 pages)
`[CHANGED]` Contributions must be HONEST:
1. An empirical evaluation of 3 open-source TSFMs on 3 industrial PdM datasets spanning forecasting, anomaly detection, and RUL tasks
2. Analysis of zero-shot vs. few-shot adaptation effectiveness using LoRA on industrial sensor data
3. A reproducible experimental framework with open-source code and public datasets

`[REMOVED]` "First industrial PdM benchmark" → say "a focused empirical study"  
`[REMOVED]` "SCADA preprocessing pipeline" as a contribution → it's standard preprocessing, describe it in methodology  
`[REMOVED]` "Federated readiness checklist" → speculative, no experiments support it  
`[REMOVED]` "Privacy leakage" claims → no privacy analysis conducted  

### Section 2: Related Work (0.7 pages)
`[CHANGED]` MUST cite Dintén & Zorrilla (2025) on TSFMs for RUL — directly relevant prior work  
`[CHANGED]` MUST cite FoundTS and TSFM-Bench — position your work as complementary (industrial focus) not competitive  
`[NEW]` Add explicit "Positioning" paragraph: "While FoundTS and TSFM-Bench provide comprehensive evaluations on general forecasting benchmarks, industrial PdM data differs in distribution, noise characteristics, and task formulation. Our work complements these benchmarks by focusing specifically on industrial sensor data."

### Section 3: Methodology (1.5 pages)
- 3.1 Datasets (Table with 3 datasets, actual statistics after download)
- 3.2 Models (Table with 4 models, capabilities)
- 3.3 Preprocessing (describe honestly — chronological splits, normalization, sequence windowing)
- 3.4 Evaluation Protocol (zero-shot, few-shot with LoRA)
- 3.5 Metrics (MAE, RMSE for forecasting; F1, AUC-ROC for anomaly; C-Index for RUL)

### Section 4: Results (2.0 pages)
`[CHANGED]` ALL numbers must come from actual experiments. No placeholders.
- 4.1 Zero-shot Performance (Table: 4 models × 3 datasets)
- 4.2 Few-shot Adaptation (Table: 2 LoRA models × 3 datasets × before/after)
- 4.3 Cross-condition Transfer (C-MAPSS FD001 → FD002/FD003/FD004 only)
- 4.4 Key Observations (data-driven analysis, not pre-written claims)

### Section 5: Discussion (0.7 pages)
`[CHANGED]` Write AFTER results. Discuss:
- Where TSFMs succeed vs. fail on industrial data
- What makes industrial PdM different from standard benchmarks
- Practical implications for PdM practitioners
- Limitations of this study

`[REMOVED]` "Taxonomy of Failures" as a formal contribution → discuss observations from your actual results  
`[REMOVED]` "Federation unreadiness" → not tested, not your contribution here  

### Section 6: Conclusion (0.3 pages)
- Summarize ACTUAL findings only
- Future work: extend to more TSFMs, federated settings (brief mention)

---

## Revised Implementation Requirements

### What Must Be Fixed/Added Before Experiments

1. **FIX PatchTST predict()** — The stub returning last-value repeat is CRITICALLY WRONG. PatchTST must be properly trained and evaluated. Use `neuralforecast` properly:
   ```python
   from neuralforecast.models import PatchTST
   model = PatchTST(h=96, input_size=512, ...)
   nf = NeuralForecast(models=[model], freq='...')
   nf.fit(df_train)
   predictions = nf.predict(df_test)
   ```

2. **ADD Lag-Llama wrapper** — Replace TimeGPT with Lag-Llama. It's open-source, lightweight, and LoRA-compatible. Use:
   ```python
   from lag_llama.gluon.estimator import LagLlamaEstimator
   ```

3. **FIX MOMENT LoRA duplication** — Remove duplicated target_modules detection (lines 73-84).

4. **REMOVE fabricated tables** — Delete all placeholder numbers from paper/tables/*.tex. These will be regenerated from real results.

5. **UPDATE MODEL_REGISTRY** — Add lag_llama, remove timegpt.

### What Can Stay As-Is
- `src/data/download.py` — functional
- `src/data/preprocessing.py` — functional (rename "SCADA pipeline" to just "preprocessing")
- `src/evaluation/metrics.py` — functional
- `src/experiments/*.py` — functional (update model lists)
- `src/visualization/*.py` — functional
- `config/*.yaml` — update to reflect reduced scope
- Notebook structure — keep 01-07, drop 08

---

## Revised Compute Estimate

| Phase | GPU Hours | Platform |
|-------|-----------|----------|
| Download + Preprocessing | 0 (CPU) | Colab Free or Local |
| MOMENT zero-shot (3 datasets) | 1-2 hrs | Colab T4 |
| Chronos zero-shot (3 datasets) | 2-3 hrs | Colab T4 |
| Lag-Llama zero-shot (3 datasets) | 1 hr | Colab T4 |
| PatchTST training + eval (3 datasets) | 2-3 hrs | Colab T4 |
| Few-shot LoRA (MOMENT + Lag-Llama) | 2-3 hrs | Colab T4 |
| Cross-condition (C-MAPSS subsets) | 1-2 hrs | Colab T4 |
| Visualization + Tables | 0 (CPU) | Local |
| **TOTAL** | **~10-15 hrs GPU** | **Colab Free Tier is sufficient** |

`[CHANGED]` **You do NOT need Colab Pro for this reduced scope.** Free tier with T4 (12hr sessions) is enough if you checkpoint properly.

---

## Revised Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 1 | Fix code (PatchTST, Lag-Llama, remove fabricated data) | Working codebase |
| Day 2 | Download datasets (C-MAPSS, MIMII, Wind SCADA) + preprocess | data/processed/ populated |
| Day 3 | Run zero-shot experiments (all 4 models × 3 datasets) | results/zero_shot/ populated |
| Day 4 | Run few-shot + cross-condition experiments | results/few_shot/ populated |
| Day 5 | Generate figures + tables from REAL results | paper/figures/ + paper/tables/ populated |
| Day 6 | Rewrite paper sections with REAL numbers | Paper v1 complete |
| Day 7 | Proofread, compile LaTeX, final check | Submission-ready PDF |

---

## What This Paper Will NOT Be

To be clear with yourself:
- This is NOT a NeurIPS-level benchmark paper
- This is NOT the "definitive" TSFM evaluation for PdM
- This is NOT going to get 100+ citations
- This IS a solid first publication establishing your research direction
- This IS an honest empirical study that contributes practical insights
- This IS a foundation for your PhD's federated learning work
