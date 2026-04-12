# Dataset, Experiment, and Model Audit

## Scope

This audit reviews the runnable experiment pipeline after replacing MIMII with PHM 2010 Milling, with emphasis on Colab feasibility, experiment validity, and publication readiness.

## 1. Dataset Decision

### Why MIMII was rejected

- The official Zenodo release is 100.2 GB in total.
- Even the single `6_dB_fan.zip` archive is about 10.2 GB compressed.
- The pipeline had to convert large 8-channel audio recordings into MFCC timelines before windowing, which is the main reason Colab RAM was exceeded.
- The paper described audio anomaly detection, but the code path treated MIMII as a generic forecasting surrogate, which weakened task fidelity.

### Why PHM Milling was selected

- It remains a public industrial PdM dataset.
- It is substantially smaller than MIMII and easier to process on free Colab.
- The raw modality is better aligned with the rest of the repository because it can be reduced to compact multivariate cut-level sequences.
- The repository already contained a PHM Milling entry in configs, references, and planning documents.

### Implementation approach

- Raw cutter CSV files are streamed in chunks.
- Each cut is summarized into compact statistical features over the 7 sensor channels.
- Windows are created at the cut level, not the raw waveform level.
- Split boundaries are respected so forecasting windows do not cross cutter trajectories.

## 2. Experiment Validity Findings

### Validated

- Zero-shot evaluation is runnable for all four current models.
- Cross-condition transfer on C-MAPSS remains a defensible experiment.
- Efficiency profiling and result aggregation are useful for deployment-oriented reporting.

### Not validated

- The current few-shot path for MOMENT is a no-op.
- The current few-shot path for Lag-Llama is a no-op.
- Reporting those runs as real adapter-based fine-tuning would be misleading.

### Corrections applied

- The runnable pipeline now uses `cmapss`, `wind_scada`, and `phm_milling`.
- The notebook mirrors the same dataset choice and preprocessing path.
- The few-shot script now fails explicitly for wrappers that do not implement real adaptation.
- The paper text was updated to describe PHM Milling instead of MIMII.
- The paper now treats few-shot as exploratory until stable adapter training is implemented.

## 3. Model Validation

| Model | Zero-shot | Few-shot | Notes |
|---|---|---|---|
| MOMENT | Usable | Not validated | Forecasting uses reconstruction-based extrapolation; adapter hook is currently a placeholder. |
| Chronos | Usable | Not applicable | Univariate model; multivariate inputs are averaged across channels before inference. |
| Lag-Llama | Usable | Not validated | Zero-shot works; adapter hook is currently a placeholder. |
| PatchTST | Usable | Usable as limited-data supervised baseline | Requires fitting before prediction; not a true zero-shot foundation model. |

## 4. Publication-Ready Contribution Framing

### Defensible contributions

1. Empirical zero-shot comparison of open-source TSFMs and a supervised baseline on three industrial datasets.
2. Cross-condition transfer study on C-MAPSS.
3. Practical deployment analysis using runtime, memory, and parameter counts.

### Claims to avoid unless new work is completed

- Do not claim validated few-shot LoRA results until adapters perform real parameter updates.
- Do not describe MIMII as part of the benchmark unless the audio preprocessing path is restored and used.
- Do not mention anomaly-detection metrics unless the pipeline truly evaluates anomaly labels.

## 5. Remaining Priorities

### Must do before submission

1. Run the updated preprocessing and zero-shot experiments on the final dataset trio.
2. Regenerate the paper tables with real PHM Milling results.
3. Decide whether to implement real few-shot adaptation or remove the few-shot section from the final manuscript.

### Should do next

1. Add explicit documentation in the paper that Chronos is evaluated through channel averaging on multivariate inputs.
2. Update the README so the advertised scope matches the runnable benchmark.
3. Add a simple temporal-split sanity check in preprocessing to guard against leakage regressions.
