# Final Review Report: TSFM-PdM Benchmark Project

**Reviewer:** Senior AI Review Agent  
**Role:** Strict senior reviewer — no compliments  
**Date:** April 10, 2026

---

## Executive Verdict

**This project is currently NOT submittable.** The codebase is ~55% structurally complete and 0% experimentally complete. The paper contains fabricated results — specific numerical claims (35% MAE degradation, 62% RUL failure, 18% preprocessing improvement) with no experimental backing. This is an academic integrity violation that must be addressed before any other work.

The underlying research question — "How well do TSFMs transfer to industrial PdM?" — is sound and appropriate for ICATH. The contribution, after honest rescoping, is sufficient for a conference proceedings paper. But the current state of the project requires significant remediation.

---

## Summary of Findings

### 1. Implementation Gap Analysis

| Category | Planned | Implemented | Gap |
|----------|---------|-------------|-----|
| Models | 9 (6 TSFM + 3 baselines) | 4 (2 TSFM + 1 stub + 1 API) | 56% missing |
| Datasets | 6 | 0 downloaded | 100% missing |
| Experiments | 3 scenarios × 9 models × 6 datasets | 0 | 100% missing |
| Paper sections | 7 | 7 (all fabricated) | 100% needs rewrite |
| Notebooks | 8 | 7 templates | 0% executed |

**Critical bugs:**
- `src/models/patchtst.py`: `predict()` returns naive last-value repeat instead of real inference — this would invalidate the primary baseline comparison
- `src/models/moment.py`: Lines 73-84 contain duplicated LoRA `target_modules` detection code
- `src/models/__init__.py`: MODEL_REGISTRY has 4 entries vs. plan's 9
- 5 model wrappers completely missing (Sundial, Time-MoE, Lag-Llama, Autoformer, Vanilla Transformer)

Full details: `plans/review/01_IMPLEMENTATION_GAP_ANALYSIS.md`

### 2. Technical Issues in the Plan

**Overclaimed contributions:**
1. ❌ "First comprehensive evaluation of TSFMs for industrial PdM" — FALSE. Dintén & Zorrilla (2025) and Jin et al. (2025) already published similar work. FoundTS (42 citations) and TSFM-Bench (42 citations) are comprehensive TSFM benchmarks.
2. ❌ "Novel SCADA preprocessing pipeline" — z-score normalization + chronological splits + Kalman imputation is standard practice in time series preprocessing.
3. ❌ "Federated Readiness Checklist" — speculative, unsubstantiated by experiments, adds no value.
4. ❌ "Privacy leakage analysis of embeddings" — no privacy analysis exists in the codebase.

**Corrected contributions (honest version):**
1. ✅ Empirical comparison of 4 TSFMs on 3 industrial PdM datasets across zero-shot and few-shot scenarios
2. ✅ Cross-condition transfer evaluation on C-MAPSS turbofan degradation data
3. ✅ Practical deployment guidelines: model efficiency (VRAM, latency, parameters) for edge/cloud PdM systems

**Dataset issues:**
- MIMII is an audio anomaly dataset — converting to TSFM-compatible time series requires mel-spectrogram extraction, which is not handled in the preprocessing code
- Wind SCADA (Kaggle) has no established train/test split convention — need to define chronological splits
- PHM Milling, PU Bearings, PRONOSTIA are in the plan but not prioritized in the reduced scope

Full details: `plans/review/02_CONTRIBUTION_EVALUATION.md`

### 3. Venue Assessment

**ICATH** (International Conference on Advanced Technologies for Humanity) is:
- Multi-disciplinary (AI, sustainability, healthcare, engineering)
- Published in MDPI Engineering Proceedings (EISSN 2673-4591)
- NOT indexed in Scopus or Web of Science
- Held at ENSA Ibn Tofaïl University, Kenitra, Morocco
- Accepts applied ML papers, surveys, and framework evaluations

**For this venue:** The reduced scope (4 models × 3 datasets × 2 scenarios) is sufficient. ICATH does not require the depth of NeurIPS or ACL. An honest empirical comparison with real numbers and practical insights will be competitive.

### 4. Optimized Scope

The optimized plan reduces the project to a feasible scope:

| Dimension | Original Plan | Optimized Plan |
|-----------|---------------|----------------|
| Models | 6 TSFM + 3 baselines | 3 TSFM + 1 baseline |
| TSFMs | MOMENT, Chronos, Sundial, Time-MoE, Lag-Llama, TimeGPT | MOMENT, Chronos, Lag-Llama |
| Baseline | PatchTST, Autoformer, Vanilla Transformer | PatchTST |
| Datasets | C-MAPSS, PHM, PU, Wind, MIMII, PRONOSTIA | C-MAPSS, Wind SCADA, MIMII |
| Scenarios | Zero-shot, Few-shot, Cross-domain | Zero-shot, Few-shot (+ cross-condition on C-MAPSS) |
| Compute | ~40+ GPU hours | ~10-15 GPU hours |
| Colab tier | Pro (possibly Pro+) | Free tier sufficient |
| Timeline | Unclear | 7 days |

Full details: `plans/review/03_OPTIMIZED_PLAN.md`

### 5. Colab Decision

**Colab Free Tier is sufficient** for the optimized scope. The 4 models total ~10-15 GPU hours on T4:
- MOMENT (385M params): ~2-3 hrs across 3 datasets + LoRA fine-tuning
- Chronos (710M params): ~3-4 hrs (largest, but inference-only)
- Lag-Llama (7M params): ~1 hr (very small model)
- PatchTST (~2M params): ~2-3 hrs (requires training from scratch)

If Colab Free disconnects frequently, upgrade to Pro ($10/mo). If you want zero friction, Pro is worth it. But it is NOT required.

---

## Mandatory Actions Before Submission

### Priority 1: Academic Integrity (DO THIS FIRST)
- [ ] Remove ALL fabricated numbers from paper (tables 3-5, abstract, intro, analysis, conclusion)
- [ ] Replace with `[TBD]` until real results are available
- [ ] Delete any claims about findings that haven't been experimentally validated

### Priority 2: Code Fixes
- [ ] Fix PatchTST `predict()` — replace stub with real neuralforecast inference
- [ ] Add Lag-Llama wrapper (`src/models/lag_llama.py`)
- [ ] Fix MOMENT duplication bug (lines 73-84)
- [ ] Update MODEL_REGISTRY and config files
- [ ] Remove TimeGPT (closed-source API, non-reproducible)

### Priority 3: Run Experiments
- [ ] Download and preprocess 3 datasets
- [ ] Execute zero-shot evaluation for all 4 models
- [ ] Execute few-shot (LoRA) for MOMENT and Lag-Llama
- [ ] Validate C-MAPSS cross-condition transfer
- [ ] Save all results as JSON/CSV

### Priority 4: Rewrite Paper
- [ ] Replace all `[TBD]` with real numbers
- [ ] Rewrite abstract, results, analysis, conclusion from scratch
- [ ] Update title to: "How Ready Are Time-Series Foundation Models for Industrial Predictive Maintenance? An Empirical Study"
- [ ] Remove "novel preprocessing pipeline" contribution claim
- [ ] Add honest positioning relative to FoundTS, TSFM-Bench, Dintén 2025
- [ ] Ensure page count is within ICATH limits (6-8 pages)

Full step-by-step guide: `plans/review/04_NEXT_STEPS_GUIDE.md`

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MIMII preprocessing fails (audio data) | Medium | Low | Replace with PHM Milling or use only 2 datasets |
| Colab disconnects during long runs | Medium | Low | Checkpoint to Drive after each model; or upgrade to Pro |
| TSFMs perform poorly on all datasets | Low | None | Negative results are valid; reframe as "not ready for PdM" |
| TSFMs perform well on all datasets | Low | None | Reframe as "promising transfer, with caveats" |
| Reviewers note FoundTS/TSFM-Bench overlap | High | Medium | Cite them explicitly; differentiate via PdM domain focus |
| Model crashes on sensor data | Medium | Low | Report N/A; discuss incompatibility as a finding |

---

## Deliverables Created in This Review

| # | File | Content |
|---|------|---------|
| 1 | `plans/review/01_IMPLEMENTATION_GAP_ANALYSIS.md` | Line-by-line code vs. plan comparison |
| 2 | `plans/review/02_CONTRIBUTION_EVALUATION.md` | Venue analysis + competitive landscape |
| 3 | `plans/review/03_OPTIMIZED_PLAN.md` | Revised scope, timeline, contributions |
| 4 | `plans/review/04_NEXT_STEPS_GUIDE.md` | Step-by-step execution instructions |
| 5 | `plans/review/05_FINAL_REVIEW_REPORT.md` | This document |

---

## Final Statement

The research question is valid. The venue is appropriate. The scope (after optimization) is feasible. The codebase provides a workable starting point.

What is NOT acceptable:
1. Fabricated results in the paper
2. Overclaimed contributions
3. A baseline (PatchTST) that doesn't actually compute predictions

Fix these three things, run the experiments, and write the paper based on what you actually observe. The contribution — an honest empirical evaluation of TSFMs for industrial PdM — is sufficient for ICATH.

**Start with Step 0 today. It requires no GPU and takes 1 hour.**

---

*End of review.*
