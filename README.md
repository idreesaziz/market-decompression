# Market Decompression

**Replication code and results for:**

> *Through the Hidden States: Synthetic Latent Supervision for Cold-Start Financial Prediction*
> Idrees Mustafa Aziz — July 2026

---

## What this paper does

Financial prediction has a cold-start problem: a newly listed asset or a post-break asset has a dense price tape but almost no trustworthy forecast labels. At a 60-hour horizon, five non-overlapping labels accumulate only after 12.5 days—the window in which documented abnormal returns are largest and a direct forecaster has the least to work with.

**Market decompression** separates the two tasks a direct learner must solve with one small label budget. It first learns to recover a hidden market state (volatility pressure, jump intensity, regime) from observed prices, supervised by *instrumented synthetic worlds* in which that state is directly observed, and only then fits a small forecast head on real labels. The **transfer field** F_b(θ; R) maps exactly where synthetic latent supervision helps and where it actively hurts, market by market.

The paper introduces the transfer field, derives six testable implications, and confirms all six—including 6.3% and 7.2% pooled MSE reductions at five labels under frozen, disjoint confirmation on Bitcoin, Ethereum, and Solana.

---

## Confirmed Results (Table 4)

| Prior | θ vector | Pooled log-skill | MSE gain | t | p | p_basket |
|-------|----------|-----------------|----------|---|---|----------|
| **A** | [0.24, 0.70, 1.90, −4.28, −4.20, 3.44] | 0.0627 [0.003, 0.123] | **6.3%** | 2.14 | 0.021 | 0.029 |
| **B** | [4.20, −4.50, 3.53, −4.50, 4.20, −4.20] | 0.0725 [0.003, 0.142] | **7.2%** | 2.14 | 0.021 | 0.010 |

*10 fresh repetitions per market, disjoint from all discovery. Primary test pools 30 market-repetition draws (BTC + ETH + SOL). p_basket is the conservative per-repetition basket average test (n=10).*

**Untouched chronological holdout (Section 8.4):** Prior A remains positive (pooled +0.039, ≈3.8%, p=0.095; BTC individually p=0.030). Prior B remains ETH-driven (ETH +0.111, p=0.047). The ten-label advantage vanishes for both, as the theory predicts (P4).

**Negative transfer confirmed:** The alternating extreme anchor θ=[4.2,−4.2,4.2,−4.2,4.2,−4.2] replicates as negative (pooled −0.036, t=−2.06), confirming P6.

---

## Repository Structure

```
market-decompression/
│
├── README.md                        ← you are here
├── requirements.txt                 ← Python dependencies
├── LICENSE
│
├── configs/
│   └── experiment.json              ← full experiment configuration
│
├── src/
│   └── scalar_reconstruction/       ← core estimation library
│       ├── cli.py                   ← command-line interface (worker, make-jobs, ...)
│       ├── eval_transfer.py         ← transfer field evaluation kernel
│       ├── jobs.py                  ← job queue management
│       ├── worker.py                ← distributed worker
│       ├── surrogate.py             ← GP surrogate utilities
│       ├── data.py                  ← chronological data splits
│       ├── skill_score.py           ← log-skill metric
│       └── ...
│
├── scripts/                         ← post-hoc analysis (run after job evaluation)
│   ├── 01_fit_surrogate.py          ← fit noise-aware ARD Matérn-2.5 GP surrogates
│   ├── 02_estimate_envelopes.py     ← envelope volume via posterior draws
│   ├── 03_select_adaptive_theta.py  ← greedy diverse adaptive θ selection
│   ├── 04_analyze_adaptive.py       ← merge adaptive results, select confirmation set
│   └── 05_generate_report.py        ← produce all tables and the final report
│
└── results/                         ← pre-computed; all paper tables and figures here
    ├── tables/
    │   ├── table4_pooled_confirmation.csv   ← Table 4 (the headline result)
    │   ├── confirmation_per_market.csv      ← per-(θ, market, b) confirmation stats
    │   ├── frozen_confirmation_set.csv      ← the 12 frozen confirmation θ
    │   ├── adaptive_theta_proposal.csv      ← 40 adaptive θ and acquisition scores
    │   ├── confirmed_claims.md              ← machine-readable confirmed claims
    │   └── full_report.md                   ← full narrative report
    ├── envelopes/
    │   ├── envelope_volumes.csv             ← Figure 4 data (V̂_b^R by market and domain)
    │   └── envelope_summary.md
    ├── surrogates/
    │   ├── gp_posterior_predictions.csv     ← posterior mean/std for all 160 seed θ
    │   └── winner_curse_table.csv           ← raw vs. shrunk top-k (Section 8.5)
    └── seed_audit/
        └── seed_fdr_tiered.csv              ← BH-FDR tiered seed results (160 θ × 3 markets × 2 budgets)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The core ML library (`mde_search`, which implements the MarketTransformer and simulator) must also be installed. It is packaged separately due to PyTorch dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e src/
```

### 2. Verify the confirmed results without re-running

All pre-computed results are in `results/`. To reproduce Table 4 and the confirmed claims directly from the result files:

```bash
python scripts/05_generate_report.py
```

This reads `results/` and `runs/final_paper_ready_run/results/` (the raw job JSONs) and reproduces the exact statistics in the paper.

### 3. Inspect the key tables

```python
import pandas as pd

# Table 4 — headline confirmation
df = pd.read_csv("results/tables/table4_pooled_confirmation.csv")
print(df[df["b"] == 5][["theta_id", "source", "pool_mean", "pool_t", "p_one_sided", "sig_5pct"]])

# Envelope volumes (Figure 4)
env = pd.read_csv("results/envelopes/envelope_volumes.csv")
print(env[env["domain"] == "full"][["market", "budget", "mean", "ci90_lo", "ci90_hi"]])

# Winner's curse (Section 8.5)
wc = pd.read_csv("results/surrogates/winner_curse_table.csv")
```

---

## Full Reproduction

The full pipeline requires running ~1,000 training jobs (160 seed + 240 adaptive + 360 confirmation + 180 test-window holdout), each involving neural network pretraining and fine-tuning. We ran these on a GCP n2-standard-16 VM; approximate wall time is 48–72 hours end to end.

### Pipeline stages

| Stage | Script / Command | Output |
|-------|-----------------|--------|
| 1. Seed jobs | `cli.py make-jobs --stage seed` | 960 job files |
| 2. Run workers | `cli.py worker --config configs/experiment.json` | `runs/.../results/*.json` |
| 3. Fit surrogates | `scripts/01_fit_surrogate.py` | GP posterior predictions |
| 4. Estimate envelopes | `scripts/02_estimate_envelopes.py` | Envelope volume CSVs |
| 5. Select adaptive θ | `scripts/03_select_adaptive_theta.py` | 40 adaptive θ |
| 6. Run adaptive jobs | `cli.py make-jobs --stage final_adaptive --eval_split val` | 240 job files |
| 7. Analyze adaptive | `scripts/04_analyze_adaptive.py` | Frozen confirmation set (12 θ) |
| 8. Run confirmation | `cli.py make-jobs --stage final_confirmation --reps 10` | 360 job files |
| 9. Run test holdout | `cli.py make-jobs --stage test_window --eval_split test --reps 10` | 180 job files |
| 10. Generate report | `scripts/05_generate_report.py` | All tables |

### Key design choices

- **Chronological split**: 60% train / 20% val / 20% test, with a 64-bar gap between splits. `eval_split=val` for all discovery and confirmation stages; `eval_split=test` only for the untouched holdout (Section 8.4).
- **GP surrogate**: ARD Matérn-2.5, `normalize_y=False`, per-point observation noise α = SE²/y_std². Candidate tier (n=2–4) gets ×2 noise inflation.
- **Envelope CI**: independent pointwise draws Z_i ~ N(μ_i, σ_i²) over 4,096 Sobol points; N=1,000 samples. Conservative lower bound (positive GP correlations would widen the true CI).
- **Winner's curse correction**: GP posterior mean vs. raw mean. Top-10 average shrinkage: +0.18–+0.29 log_skill.
- **Test split**: never touched during discovery, surrogate fitting, or confirmation. The `eval_transfer.py` kernel enforces this: `support_pool = splits['train']` regardless of `eval_split`.

---

## Statistical Protocol

The discovery/confirmation/holdout architecture follows the data-snooping discipline of White (2000) and Sullivan, Timmermann & White (1999):

1. **Seed stage**: 160 θ evaluated at val split → GP surrogate fitted → no claims made
2. **Adaptive stage**: 40 θ proposed by surrogate posterior → evaluated at val split → no claims made
3. **Frozen confirmation**: 12 θ frozen before confirmation begins → 10 fresh reps each at val split → primary evidence begins here
4. **Untouched holdout**: same 3 θ (2 confirmed + 1 negative anchor) → 10 fresh reps at test split → never touched by any earlier stage

Discovery-stage statistics (raw means, GP heatmaps, envelope volumes from seed data) are **never** reported as evidence of positive transfer. Evidence begins only at Step 3.

---

## Data

The paper uses publicly available hourly close prices for BTCUSDT, ETHUSDT, and SOLUSDT from Binance. Each series spans approximately 2.8 years (≈25,000 hourly bars). Data files are not included in this repository due to size; they can be downloaded from [Binance historical data](https://data.binance.vision/) and placed in `data/`.

Expected file format:
```
data/
├── BTCUSDT_1h.csv   # columns: timestamp, open, high, low, close, volume
├── ETHUSDT_1h.csv
└── SOLUSDT_1h.csv
```

---

## Citation

```bibtex
@article{aziz2026decompression,
  title   = {Through the Hidden States: Synthetic Latent Supervision for
             Cold-Start Financial Prediction},
  author  = {Aziz, Idrees Mustafa},
  year    = {2026},
  month   = {July},
  note    = {Working paper, Middle East Technical University},
  url     = {https://github.com/idreesaziz/market-decompression}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

Code in `src/scalar_reconstruction/` implements the estimation pipeline described in Section 6 of the paper. The `mde_search` library (MarketTransformer architecture and simulator) is a dependency and is licensed separately.
