"""
Noise-aware GP surrogate fitting (Section 6.3 / Section 8.1 of the paper).

Fits one ARD Matérn-2.5 GP per (market, budget) — 6 GPs total — using
per-point observation noise estimated from bootstrap standard errors.
Candidate-tier points (n=2–4) receive doubled noise to reflect brittle SE estimates.
Produces posterior mean / std predictions (Equation 21) and a winner's-curse table
comparing raw means to shrinkage-adjusted posterior means (Section 8.5).

Inputs
------
results/seed_audit/seed_fdr_tiered.csv   — BH-FDR tiered seed evaluation results

Outputs
-------
results/surrogates/gp_posterior_predictions.csv   — posterior predictions (960 rows)
results/surrogates/winner_curse_table.csv          — raw vs. shrunk top-k table

Reference: Aziz (2026), "Through the Hidden States," Sections 5 and 6.3.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

PROJECT_ROOT = Path(__file__).parent.parent
AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "seed_audit"
)
OUT_DIR = PROJECT_ROOT / "results" / "surrogates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_MIN, THETA_MAX = -4.5, 4.5
THETA_COLS = [f"theta_{j+1}" for j in range(6)]
MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BUDGETS = [5, 10]
MIN_NOISE_S = 1e-6         # floor for per-point noise in normalised y-scale
CANDIDATE_NOISE_FACTOR = 2.0  # extra noise multiplier for n=2-4 groups
SEP = "=" * 68


# ------------------------------------------------------------------ #
# GP helpers
# ------------------------------------------------------------------ #

def normalize_theta(X_raw: np.ndarray) -> np.ndarray:
    return (X_raw - THETA_MIN) / (THETA_MAX - THETA_MIN)


def per_point_alpha(se: np.ndarray, tier: np.ndarray, y_std: float) -> np.ndarray:
    """Per-point observation noise in the normalised y-scale."""
    alpha_raw = np.where(np.isfinite(se) & (se > 0), se ** 2, 0.0)
    alpha_raw = np.where(tier == "candidate", alpha_raw * CANDIDATE_NOISE_FACTOR, alpha_raw)
    fallback = float(np.nanmedian(alpha_raw[alpha_raw > 0])) if np.any(alpha_raw > 0) else 0.01
    alpha_raw = np.where(alpha_raw <= 0, fallback, alpha_raw)
    return np.maximum(alpha_raw / max(y_std ** 2, 1e-12), MIN_NOISE_S)


def fit_gp(X_raw: np.ndarray, y: np.ndarray, se: np.ndarray, tier: np.ndarray,
           label: str = "") -> tuple:
    """Fit noise-aware ARD GP. Returns (gp, y_mean, y_std, converged, ls)."""
    X = normalize_theta(X_raw)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    y_s = (y - y_mean) / y_std
    alpha_s = per_point_alpha(se, tier, y_std)

    kernel = ConstantKernel(1.0, (1e-3, 10.0)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(0.05, 5.0),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha_s, normalize_y=False,
        n_restarts_optimizer=5, random_state=42,
    )
    converged = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gp.fit(X, y_s)
        except Exception as exc:
            print(f"  WARNING: fit failed for {label}: {exc}")
            converged = False

    k = gp.kernel_
    try:
        ls = np.asarray(k.k2.length_scale, dtype=float)
    except AttributeError:
        ls = np.asarray(k.length_scale, dtype=float)

    return gp, y_mean, y_std, converged, ls


def predict(gp, y_mean: float, y_std: float,
            X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) in original y-scale."""
    X = normalize_theta(X_raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_s, sig_s = gp.predict(X, return_std=True)
    return mu_s * y_std + y_mean, sig_s * y_std


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    print("Loading tiered FDR data...")
    tiered = pd.read_csv(AUDIT_DIR / "seed_fdr_tiered.csv")
    print(f"  {len(tiered)} rows  |  {tiered['theta_id'].nunique()} unique theta_ids")
    print(f"  markets: {sorted(tiered['market'].unique())}")
    print(f"  budgets: {sorted(tiered['b'].unique())}")

    gp_store: dict[tuple, tuple] = {}   # (mkt, b) -> (gp, y_mean, y_std, converged)
    pred_rows: list[dict] = []

    print(f"\n{SEP}")
    print("FITTING SURROGATES")
    print(SEP)

    for mkt in MARKETS:
        for b in BUDGETS:
            sub = (tiered[(tiered["market"] == mkt) & (tiered["b"] == b)]
                   .copy()
                   .reset_index(drop=True))
            if sub.empty:
                print(f"  WARNING: no data for {mkt} b={b}")
                continue

            X_raw = sub[THETA_COLS].to_numpy(float)
            y = sub["mean"].to_numpy(float)
            se = sub["se"].to_numpy(float)
            tier = sub["evidence_tier"].to_numpy(str)

            gp, y_mean, y_std, converged, ls = fit_gp(
                X_raw, y, se, tier, label=f"{mkt} b={b}"
            )
            gp_store[(mkt, b)] = (gp, y_mean, y_std, converged)

            mu, sigma = predict(gp, y_mean, y_std, X_raw)
            prob_pos = norm.cdf(mu / np.maximum(sigma, 1e-9))

            ls_str = " ".join(f"{v:.2f}" for v in ls)
            print(
                f"  {mkt} b={b:2d} | n={len(sub):3d} | ls=[{ls_str}]"
                f" | y_std={y_std:.4f} | converged={converged}"
            )

            for idx in range(len(sub)):
                row = sub.iloc[idx]
                is_confirmed = bool(
                    row["evidence_tier"] == "anchor"
                    and row.get("sig_5pct_FDR_tiered", False)
                )
                pred_rows.append({
                    "market": mkt,
                    "budget": b,
                    "theta_id": row["theta_id"],
                    **{c: row[c] for c in THETA_COLS},
                    "design_source": row["design_source"],
                    "evidence_tier": row["evidence_tier"],
                    "n_reps": int(row["n"]),
                    "observed_mean_log_skill": float(row["mean"]),
                    "observed_se_log_skill": float(row["se"]),
                    "observed_ci95_lo": float(row["ci95_lo"]),
                    "observed_ci95_hi": float(row["ci95_hi"]),
                    "prob_gt0_bootstrap": float(row["prob_gt0"]),
                    "posterior_mean": float(mu[idx]),
                    "posterior_std": float(sigma[idx]),
                    "prob_positive": float(prob_pos[idx]),
                    "shrinkage": float(row["mean"] - mu[idx]),
                    "is_confirmed_anchor": is_confirmed,
                    "model_used": "GP_ARD_Matern25",
                    "fit_converged": converged,
                })

    pred_df = pd.DataFrame(pred_rows)
    out_pred = OUT_DIR / "gp_posterior_predictions.csv"
    pred_df.to_csv(out_pred, index=False)
    print(f"\nSaved: {out_pred}  ({len(pred_df)} rows)")

    # ------------------------------------------------------------------ #
    # Winner's curse / shrinkage table
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("WINNER'S CURSE ANALYSIS")
    print(SEP)

    shrunk_rows: list[pd.DataFrame] = []
    for mkt in MARKETS:
        for b in BUDGETS:
            sub = pred_df[
                (pred_df["market"] == mkt) & (pred_df["budget"] == b)
            ].copy().reset_index(drop=True)

            sub = sub.sort_values("observed_mean_log_skill", ascending=False)
            sub["rank_by_raw_mean"] = range(1, len(sub) + 1)

            sub_post = sub.sort_values("posterior_mean", ascending=False)
            sub_post = sub_post.reset_index(drop=True)
            sub_post["rank_by_posterior_mean"] = range(1, len(sub_post) + 1)

            merged = sub.merge(
                sub_post[["theta_id", "rank_by_posterior_mean"]], on="theta_id", how="left"
            )
            top25_raw = set(sub.head(25)["theta_id"])
            top25_post = set(sub_post.head(25)["theta_id"])
            merged["in_top25_raw"] = merged["theta_id"].isin(top25_raw)
            merged["in_top25_posterior"] = merged["theta_id"].isin(top25_post)
            merged["in_any_top25"] = merged["in_top25_raw"] | merged["in_top25_posterior"]

            in_any = merged[merged["in_any_top25"]].copy()
            shrunk_rows.append(in_any)

            top10 = merged[merged["in_top25_raw"]].head(10)
            mean_shrink = float(top10["shrinkage"].mean())
            max_shrink = float(top10["shrinkage"].max())
            print(
                f"  {mkt} b={b} | top-10 raw mean shrinkage: avg={mean_shrink:+.4f}"
                f"  max={max_shrink:+.4f}"
            )

    shrunk_df = pd.concat(shrunk_rows, ignore_index=True)
    out_shrunk = OUT_DIR / "winner_curse_table.csv"
    shrunk_df.to_csv(out_shrunk, index=False)
    print(f"\nSaved: {out_shrunk}  ({len(shrunk_df)} rows)")

    # ------------------------------------------------------------------ #
    # Confirmed anchors with posterior stats
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("CONFIRMED ANCHORS WITH POSTERIOR MEANS")
    print(SEP)
    confirmed = pred_df[pred_df["is_confirmed_anchor"]].sort_values(
        "posterior_mean", ascending=False
    )
    for _, r in confirmed.iterrows():
        th = [round(r[c], 2) for c in THETA_COLS]
        print(
            f"  {r['market']} b={int(r['budget'])}"
            f" | raw={r['observed_mean_log_skill']:+.3f}"
            f" -> posterior={r['posterior_mean']:+.3f}"
            f" | shrink={r['shrinkage']:+.3f}"
            f" | prob_pos={r['prob_positive']:.3f}"
            f" | theta={th}"
        )

    print(f"\n{SEP}")
    print("DONE")
    print(f"  {out_pred}")
    print(f"  {out_shrunk}")
    print(SEP)


if __name__ == "__main__":
    main()
