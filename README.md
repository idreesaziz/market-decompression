<div align="center">

# Market Decompression

**Replication code and pre-computed results for**

*Through the Hidden States: Synthetic Latent Supervision for Cold-Start Financial Prediction*

Idrees Mustafa Aziz · July 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

</div>

---

## Overview

Financial prediction has a cold-start problem: at a 60-hour horizon, only five non-overlapping labels accumulate in the first 12 days — the window where documented abnormal returns are largest and a direct forecaster has the least to work with.

**Market decompression** decouples the two tasks a direct learner must solve with a small label budget. A hidden market state (volatility pressure, jump intensity, regime) is first recovered from prices using *instrumented synthetic worlds*, where that state is directly observable. A small forecast head then trains on those recovered states rather than raw prices. The **scalar transfer field** F_b(θ; R) maps exactly where synthetic supervision helps and where it actively hurts — market by market, budget by budget.

This repository contains the full estimation pipeline, all pre-computed results, and the code to verify every number in the paper without re-running jobs.

---

## Results

### Table 4 — Confirmed Transfer at b = 5 Labels

| Prior | θ | Pooled log-skill | MSE gain | t | p |
|:---:|:---|:---:|:---:|:---:|:---:|
| **A** | [0.24, 0.70, 1.90, −4.28, −4.20, 3.44] | 0.0627 [0.003, 0.123] | **6.3%** | 2.14 | 0.021 |
| **B** | [4.20, −4.50, 3.53, −4.50, 4.20, −4.20] | 0.0725 [0.003, 0.142] | **7.2%** | 2.14 | 0.021 |

*10 fresh i.i.d. repetitions per market (BTC · ETH · SOL), disjoint from all discovery. Pooled across 30 market-rep draws. p one-sided.*

**Untouched holdout (Section 8.4):** Prior A remains positive (pooled +0.039, ~3.8%, p=0.095; BTC individually p=0.030). Prior B remains ETH-driven (ETH +0.111, p=0.047). The 10-label advantage vanishes for both — consistent with P4.

**Negative transfer confirmed (Section 8.8):** Alternating extreme anchor θ=[4.2,−4.2,4.2,−4.2,4.2,−4.2] replicates as negative (pooled −0.036, t=−2.06), confirming P6.

---

## Quickstart

### Verify pre-computed results (no re-running needed)

```bash
pip install -r requirements.txt
pip install -e src/
python scripts/00_verify_results.py
```

This prints every number in the paper directly from the CSVs in `results/`.

### Inspect tables in Python

```python
import pandas as pd

# Table 4 — headline confirmation
pool = pd.read_csv("results/tables/table4_pooled_confirmation.csv")
print(pool[pool["b"] == 5][["theta_id", "pool_mean", "pool_t", "p_one_sided", "sig_5pct"]])

# Envelope volumes — Figure 4
env = pd.read_csv("results/envelopes/envelope_volumes.csv")
print(env[env["domain"] == "full"][["market", "budget", "mean", "ci90_lo", "ci90_hi"]])

# Winner's curse table — Section 8.5
wc = pd.read_csv("results/surrogates/winner_curse_table.csv")
```

---

## Repository Structure

```
market-decompression/
│
├── scripts/
│   ├── 00_verify_results.py          verify all paper numbers from pre-computed CSVs
│   ├── 01_fit_surrogate.py           noise-aware ARD Matérn-2.5 GP per (market, budget)
│   ├── 02_estimate_envelopes.py      envelope volume via 500 posterior draws
│   ├── 03_select_adaptive_theta.py   GP-guided adaptive θ selection (40 points)
│   ├── 04_analyze_adaptive.py        merge adaptive results → frozen confirmation set
│   └── 05_generate_report.py         full tables + confirmed_claims.md
│
├── src/scalar_reconstruction/        core estimation library
│   ├── cli.py                        make-jobs / worker CLI
│   ├── eval_transfer.py              transfer field evaluation kernel
│   ├── surrogate.py                  GP surrogate utilities
│   ├── data.py                       chronological data splits
│   ├── skill_score.py                log-skill metric
│   └── ...
│
├── results/
│   ├── tables/
│   │   ├── table4_pooled_confirmation.csv    Table 4 (headline result)
│   │   ├── confirmation_per_market.csv       per-(θ, market, b) stats
│   │   ├── frozen_confirmation_set.csv       12 frozen confirmation θ
│   │   ├── adaptive_theta_proposal.csv       40 adaptive θ + acquisition scores
│   │   ├── confirmed_claims.md               safe confirmed claims only
│   │   └── full_report.md                    full narrative report
│   ├── envelopes/
│   │   ├── envelope_volumes.csv              Figure 4 data
│   │   └── envelope_summary.md
│   ├── surrogates/
│   │   ├── gp_posterior_predictions.csv      posterior mean/std for 160 seed θ
│   │   └── winner_curse_table.csv            raw vs. shrunk top-k (Section 8.5)
│   └── seed_audit/
│       └── seed_fdr_tiered.csv               BH-FDR tiered seed results
│
├── configs/experiment.json           full experiment configuration
├── data/README.md                    data format + Binance download instructions
├── requirements.txt
└── LICENSE
```

---

## Full Reproduction

The complete pipeline runs ~1,000 training jobs (160 seed + 240 adaptive + 360 confirmation + 180 test holdout), each involving neural network pretraining and fine-tuning. We ran on a GCP n2-standard-16 VM; total wall time is approximately 48–72 hours.

| # | Stage | Command | Output |
|:---:|---|---|---|
| 1 | Seed jobs | `cli.py make-jobs --stage seed` | 960 job files |
| 2 | Run workers | `cli.py worker --config configs/experiment.json` | `runs/*/results/*.json` |
| 3 | Fit surrogates | `scripts/01_fit_surrogate.py` | GP posterior predictions |
| 4 | Estimate envelopes | `scripts/02_estimate_envelopes.py` | Envelope volume CSVs |
| 5 | Select adaptive θ | `scripts/03_select_adaptive_theta.py` | 40 adaptive θ |
| 6 | Run adaptive jobs | `cli.py make-jobs --stage final_adaptive --eval_split val` | 240 job files |
| 7 | Analyze adaptive | `scripts/04_analyze_adaptive.py` | Frozen confirmation set |
| 8 | Run confirmation | `cli.py make-jobs --stage final_confirmation --reps 10` | 360 job files |
| 9 | Run test holdout | `cli.py make-jobs --stage test_window --eval_split test --reps 10` | 180 job files |
| 10 | Generate report | `scripts/05_generate_report.py` | All tables |

### Key design choices

- **Chronological split** — 60% train / 20% val / 20% test with a 64-bar gap between splits. `eval_split=val` throughout discovery and confirmation; `eval_split=test` only for the untouched holdout.
- **GP surrogate** — ARD Matérn-2.5, `normalize_y=False`, per-point noise α = SE²/y_std². Candidate tier (n=2–4) receives ×2 noise inflation.
- **Winner's curse correction** — GP posterior mean vs. raw mean. Top-10 average shrinkage: +0.18 to +0.29 log-skill.
- **Envelope CI** — 500 independent posterior function draws over a 16,384-point Sobol grid. Conservative lower bound (positive GP correlations would widen the true CI).
- **Test split discipline** — `eval_transfer.py` sets `support_pool = splits['train']` unconditionally; normalization statistics are always fitted on training indices only.

---

## Statistical Protocol

The architecture follows the data-snooping discipline of White (2000):

1. **Seed stage** — 160 θ evaluated on val split → GP surrogate fitted → no claims made
2. **Adaptive stage** — 40 θ proposed by GP posterior → evaluated on val split → no claims made
3. **Frozen confirmation** — 12 θ frozen before confirmation begins → 10 fresh reps on val → primary evidence
4. **Untouched holdout** — 3 θ (2 confirmed + 1 negative anchor) → 10 fresh reps on test → never touched by any earlier stage

Discovery-stage statistics (raw means, GP heatmaps, envelope volumes) are never reported as evidence of transfer. Evidence begins only at Step 3.

---

## Data

Hourly OHLCV for BTCUSDT, ETHUSDT, and SOLUSDT from Binance (~25,000 bars per market, 2021–2024). Not included here due to size. See [`data/README.md`](data/README.md) for the download script and expected file format.

---

## Citation

```bibtex
@article{aziz2026decompression,
  title   = {Through the Hidden States: Synthetic Latent Supervision for
             Cold-Start Financial Prediction},
  author  = {Aziz, Idrees Mustafa},
  year    = {2026},
  month   = {July},
  note    = {Working paper},
  url     = {https://github.com/idreesaziz/market-decompression}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
