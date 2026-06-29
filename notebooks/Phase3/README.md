# Phase 3 — Association Rule Mining

**Owner:** Pattern Analyst &nbsp;|&nbsp; **Dataset:** PaySim Synthetic Financial Fraud &nbsp;|&nbsp; **Method:** Apriori (frequent itemsets → association rules)

## Goal of this phase
Surface **co-occurrence patterns** between transaction attributes that are not obvious from
simple tabulation — expressed as `IF antecedent THEN consequent` rules and ranked by
**Support, Confidence, and Lift**. The phase deliberately follows the project's
knowledge-discovery goal (*interpretation of hidden patterns, not prediction accuracy*).

## How this fits the KDD pipeline
Phase 1 (Data Engineer) built a preprocessing pipeline with two output branches:

- **segmentation branch** → scaled numeric matrix → **Phase 2 (Clustering)**
- **pattern branch** (`select_pattern_features`) → 3 discretized categorical columns
  (`type`, `time_segment`, `amount_category`) saved as `data_phase3_rules.parquet`
  → **this phase**

This phase then feeds **Phase 4 (Anomaly Detection)** with a behavioural anomaly flag
(see *Hand-off to Phase 4* below).

> **Label rule (important):** the Dataset Reference Document states that PaySim's `isFraud`
> label **must not be used as a classification target during any phase** — it may only be
> **consulted after mining to validate** discovered patterns. Both notebooks obey this: all
> Apriori mining is done on attributes only, and `isFraud` is touched solely in a clearly
> separated post-mining validation step.

---

## Notebooks

### 1. `PA_association_rule_mining.ipynb` — core deliverable (label-free)
Mines the 3-column Phase 1 hand-off and discovers the **normal structure** of transactions.

Pipeline: load hand-off → review discretization → one-hot encode (11 items) → Apriori
(`min_support = 0.01`) → association rules → filter to non-trivial high-lift rules → document.

**Result:** 92 candidate rules → 15 strong → **11 documented rules** (lift 1.4–2.7), each with
business commentary. Headline finding: *transaction `type` strongly predetermines transaction
size* (e.g. `TRANSFER → High_Amount`, conf 82%, lift 2.45).

Covers the rubric: **Discretization**, **Rule Generation** (Support/Confidence/Lift),
**Rule Interpretation** (≥10 non-trivial rules).

### 2. `PA_anomaly_pattern_mining_and_validation.ipynb` — extension (label-free mining + post-mining validation)
Re-derives a richer feature set from the raw PaySim log (adding balance-behaviour signals:
drain ratio, emptied-origin, balance consistency, destination kind) and mines **association
rules on those behavioural attributes only — `isFraud` is never a mining input.** After mining,
a separate validation step consults `isFraud` *only* to measure which discovered patterns align
with real fraud (which the dataset rule explicitly permits). This is the bridge into Phase 4.

**Result (run on the full 6,362,620 transactions; fraud baseline 0.129%):**
- 1,914 frequent behavioural itemsets → 2,022 behavioural association rules.
- **Headline discovered pattern:** `emptied_origin=Emptied  +  orig_balance_consistency=Orig_Consistent`
  (an account drained to exactly zero with perfectly reconciling books) — a rare, distinctive
  behavioural signature found **without the label**. Post-mining validation shows it (and its
  refinements with `Full_Drain` / `TRANSFER` / `CASH_OUT`) aligns with fraud at a **100% fraud
  rate**.
- **Behavioural anomaly flag** (built from the 44 patterns validated at fraud_rate ≥ 0.5):
  **70.1% precision, 98.1% recall** (8,053 of 8,213 frauds caught, ~11.5k transactions flagged).

---

## Output files (deliverables)

| File | Produced by | Contents |
|---|---|---|
| `phase3_association_rules.csv` | notebook 1 | 11 documented structural rules + interpretation |
| `phase3_validated_anomaly_patterns.csv` | notebook 2 | discovered behavioural patterns ranked by post-mining fraud alignment |
| `phase3_rule_based_anomaly_flags.parquet` | notebook 2 | per-transaction behavioural anomaly flag + `isFraud`, for Phase 4 cross-referencing |

---

## How to run

**Environment:** use the **Anaconda Python kernel** (it has `pandas`, `pyarrow`, `scikit-learn`,
and `mlxtend`). The Apriori library is `mlxtend`; install with `pip install mlxtend` if missing.
Keep `numpy < 2` in this environment (compiled libs such as `pyarrow` require it).

**Notebook 1** runs as-is — it reads `data_phase3_rules.parquet` (the loader searches several
candidate paths automatically).

**Notebook 2** needs the raw file:
1. Download from https://www.kaggle.com/datasets/ealaxi/paysim1
2. Place `PS_20174392719_1491204439457_log.csv` into `data/raw/`
3. Re-run all cells.

---

## Hand-off to Phase 4 (Anomaly & Outlier Detection)

Phase 3 gives Phase 4 a **third independent fraud signal**. The recommended Phase 4 workflow:

1. Build IQR + Z-score detectors on the numeric features from Phase 1
   (`amount`, `errorBalanceOrig`, `balance_drain_ratio`).
2. Run **Isolation Forest** on the same numeric feature set.
3. **Cross-reference** four signals per transaction:
   `[ behavioural anomaly flag | Isolation Forest | IQR/Z-score | Phase 2 DBSCAN cluster outlier ]`.
4. Classify each anomaly as **data error / rare-but-legit / risk signal**, using agreement
   across methods as evidence — a record flagged by several independent methods is almost
   certainly a genuine risk signal; one flagged by only a single method is more likely a data
   error or a rare legitimate case.

This directly addresses the Phase 4 rubric's **Cross-referencing** and **Business
Interpretation** criteria.

### Using the hand-off file (`phase3_rule_based_anomaly_flags.parquet`)
The file is **row-aligned to the raw PaySim CSV** (same order, index `0..6,362,619`), so Phase 4
can join it to its own dataframe by position. Schema:

| Column | Meaning |
|---|---|
| `behavioural_anomaly_flag` | 1 = transaction matches a discovered, fraud-validated behavioural signature |
| `isFraud` | the true label — **for validation/cross-referencing only, never as a model input** |

Copy-paste starting point for Phase 4:

```python
import pandas as pd

# load raw data in its original order + the Phase 3 hand-off
df  = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv')
p3  = pd.read_parquet('notebooks/Phase3/phase3_rule_based_anomaly_flags.parquet')
df['rule_flag'] = p3['behavioural_anomaly_flag'].values      # row-aligned

# ... after you compute your own detectors, cross-reference them:
# df['iso_flag']  = isolation_forest_prediction        # 1 = anomaly
# df['iqr_flag']  = iqr_or_zscore_outlier              # 1 = outlier
# agreement = df[['rule_flag','iso_flag','iqr_flag']].sum(axis=1)
# df['confidence_tier'] = agreement   # 3 = all methods agree -> strongest risk signal
```

The validated patterns themselves (with their fraud_rate) live in
`phase3_validated_anomaly_patterns.csv` — use them as the **hypotheses** your statistical
detectors should confirm.

---

## Key terms (quick reference)
- **Support** — how often the antecedent + consequent occur together (coverage).
- **Confidence** — P(consequent | antecedent); the rule's reliability.
- **Lift** — confidence ÷ baseline support of the consequent. **Lift > 1** = positive
  association; the higher, the more surprising / non-trivial the rule.
