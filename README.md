# Relay-Local Attack Detection for Power Grids on FPGA

A fixed-point logistic regression classifier that distinguishes cyber attacks from
natural faults on a power transmission system, implemented in FPGA fabric so that
each protective relay can classify its own measurements locally. Four relay-local
classifiers are combined with a one-bit consensus rule, which recovers most of the
accuracy of a centralized model while sending four bits across the network instead
of thirty-six measurements.

This extends a prior research project on the same dataset. That work asked what the
smallest set of PMU features is that a single relay could compute online at 120 Hz,
and whether a model trained on three relays generalizes to a fourth. The honest
answer was that a single relay does not have enough context for reliable detection,
and that grids would need multi-relay consensus. This project builds and measures
that consensus architecture in hardware.

## Problem

The dataset models a 3-bus, 2-line transmission system with four protective relays,
each containing a Phasor Measurement Unit sampling at 120 Hz. Distance relays trip
on apparent impedance, so an attacker who wants to cause a spurious trip spoofs
exactly that signal. The detection task is to separate genuine faults from injected
data using only what a relay can measure.

The constraint that makes this a hardware problem is the sample period. At 120 Hz a
relay has 8.33 ms between measurements, and any detection that cannot finish inside
that window is not usable for protection.

## Approach

Each relay runs an identical 9-input logistic regression over a reduced feature set:
eight PMU-derived features plus a binary flag marking an open breaker. Inference is
a dot product followed by a sign test, which is nine multiply-accumulates in
hardware. No sigmoid is required, since the sigmoid is monotonic and the decision
threshold maps to a sign test on the raw score.

Gradient boosting was evaluated as an alternative and rejected on measured grounds.
At full size it scores higher than logistic regression (0.5150 against 0.4732 macro
F1 under grouped cross-validation), but at a size that fits the fabric as
combinational logic (50 trees, depth 4) it drops to 0.4299, below the linear model.
The accuracy advantage of boosting is not available at deployable scale on this task.

## Results

All figures are macro F1. The floor is the always-predict-Attack baseline, which for
a class balance of p equals p / (1 + p).

### Held-out test set

Three recording sessions (data6, data11, data15) were held out by source file before
any model selection and scored once. 15,494 snapshots, attack rate 0.7018.

| Method | Accuracy | Precision | Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Any relay says Attack (OR, 4 bits) | 0.5614 | 0.7242 | 0.6058 | 0.5214 |
| Mean of relay scores | 0.5277 | 0.7303 | 0.5185 | 0.5080 |
| Majority vote, tie to Attack | 0.5201 | 0.7158 | 0.5243 | 0.4966 |
| Relay R4 alone | 0.5145 | 0.7176 | 0.5081 | 0.4945 |
| Relay R1 alone | 0.5128 | 0.7147 | 0.5090 | 0.4921 |
| Relay R3 alone | 0.4950 | 0.7230 | 0.4547 | 0.4845 |
| Relay R2 alone | 0.4932 | 0.7214 | 0.4525 | 0.4827 |
| Majority vote, tie to Natural | 0.4733 | 0.7171 | 0.4123 | 0.4674 |
| Unanimous Attack (AND) | 0.4606 | 0.7173 | 0.3820 | 0.4575 |
| Always Attack (floor) | 0.7018 | 0.7018 | 1.0000 | 0.4124 |

### Consensus against centralization

| Configuration | Inputs on the wire | Macro F1 | Gain over floor |
| --- | --- | --- | --- |
| Floor | none | 0.4124 | 0.0000 |
| Best single relay | 9 local features | 0.4945 | 0.0821 |
| OR consensus, 4 relays | 4 bits | 0.5214 | 0.1090 |
| Centralized, all relays | 32 features + 4 flags | 0.5724 | 0.1600 |

The OR consensus recovers 68.1 percent of the centralized model's gain over the
floor. The centralized model needs all thirty-two measurements plus four flags from
four relays at every snapshot. The consensus rule needs one bit per relay.

### Why OR and not majority vote

Requiring relays to agree makes detection worse, and the vote threshold sweep is
monotone in the held-out data:

| Relays that must say Attack | Macro F1 |
| --- | --- |
| 1 (OR) | 0.5214 |
| 2 (majority) | 0.4966 |
| 3 | 0.4674 |
| 4 (AND) | 0.4575 |

An attack on this topology is a local event. The relay nearest to it sees a clear
signature and the other three barely register it, so demanding confirmation discards
the signal. Majority vote gains essentially nothing over the best individual relay
(0.4966 against 0.4945). Lowering the bar to one relay is what works.

The combiner and vote threshold were selected on training data by sweeping all four
thresholds, then committed to this repository before the held-out set was scored. The
rule that won on training also won on the holdout, with the ranking of all nine
methods preserved except for a swap between R2 and R3.

## Hardware architecture

```
  relay 1 features --> [ LR core 1 ] --+
  relay 2 features --> [ LR core 2 ] --+
                                       +--> OR --> attack / natural
  relay 3 features --> [ LR core 3 ] --+
  relay 4 features --> [ LR core 4 ] --+
```

Each LR core is a 9-input fixed-point dot product, a bias add, and a sign test,
producing one bit. The consensus layer is a single OR gate. Four cores and the gate
target one Zynq-7020 (Pynq-Z2).

The hardware and software boundary is drawn deliberately:

- Host (ARM processing system, Python): read features, cap `PA:Z` at the training
  quantile, derive the `PA:Z__open` flag, standardize to z-scores, quantize, and pack
  the DMA buffer.
- Fabric: four dot products, four sign tests, one OR gate.

Standardization and signal conditioning are fixed affine operations that in a real
relay would live in the analog front end, not in the classifier. Keeping them on the
host means the fabric implements the classifier and nothing else.

## Evaluation protocol

Rows in this dataset are 99.92 percent autocorrelated in contiguous blocks, so a
random train/test split places near-duplicate neighbors on both sides and inflates
every metric. Published results on this dataset that use a random split are not
comparable to the numbers above.

Every figure here uses `StratifiedGroupKFold` grouped by source file, so an entire
recording session is either wholly in training or wholly in test. The held-out set is
three whole source files.

Feature selection was re-derived on the twelve training files alone, after the holdout
was carved out, using SHAP rankings averaged over five folds rather than a single
fold. Three features rank identically in all five folds (`PM2:V`, `PM1:V`, `PM5:I`).
Ranks six through ten are a statistical tie, so the tail of the selected set is not
stable and is reported as such.

## Repository layout

```
notebooks/
  power_grid_classification.ipynb   original research: EDA, leakage analysis,
                                    feature importance, leave-one-relay-out
  model_exports.ipynb               feature re-derivation, consensus experiment,
                                    held-out evaluation, weight freeze
models/
  lr_weights.npz                    frozen coefficients, scaler parameters,
                                    PA:Z cap, feature order, vote threshold
  holdout_vectors.npz               held-out feature matrices and expected
                                    outputs, for hardware replay
python/
  lr_inference.py                   integer-only reference implementation
data/
  power_system_combined.csv         not tracked, see Data below
```

## Data

Mississippi State University and Oak Ridge National Laboratory Power System Attack
Dataset, originally fifteen CSV files of roughly 5,000 rows each. The combined file is
78,377 rows by 130 columns and is not tracked in this repository because of its size.

Source: https://www.kaggle.com/datasets/bachirbarika/power-system/data

Concatenate the fifteen files from `binaryAllNaturalPlusNormalVsAttacks/` with a
`source_file` provenance column and save as `data/power_system_combined.csv`.

## Reproducing

```
pip install numpy pandas scikit-learn xgboost shap matplotlib pytest
jupyter lab notebooks/model_exports.ipynb
```

Run all cells in order. Every random seed is fixed at 42. The notebook asserts the
expected data shape, block structure, and group separation, so a mismatch fails
loudly rather than producing quietly wrong numbers.

## Project status

Complete:

- Model frozen and verified. The saved weights reproduce the scikit-learn decision
  function with a maximum error of 0.0 across all 61,976 held-out relay rows.
- Consensus architecture selected, pre-registered, and validated once on held-out data.

In progress:

- Fixed-point quantization analysis and format selection.
- Vitis HLS kernel, C simulation, and C/RTL co-simulation.
- Vivado block design, bitstream, and deployment to Pynq-Z2.
- Measured latency and resource utilization.
- Hardware replay of the held-out set.

## Limitations

Absolute performance is low. The best configuration reaches 0.5214 macro F1 against a
floor of 0.4124. The finding is not that this detector is accurate. It is that
relay-local detection has an information ceiling that model capacity does not fix, and
that a one-bit consensus rule is the cheapest way found here to push past it.

Other limitations carried from the research phase: labels are system-level and
inherited by all four relay views, so a remote relay is labeled Attack for a
disturbance it may barely see. The infinity cap quantile of 0.999 was chosen rather
than tuned. Results come from a single testbed topology. No hyperparameter tuning was
performed, deliberately, since tuning under a leaky protocol optimizes for the leak.
