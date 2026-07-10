"""
GP posterior envelope volume estimation (Section 7 / Figure 4).

For each (market, budget) GP fitted in script 01, samples 500 posterior
function realisations over a 16384-point Sobol grid in [-4.5, 4.5]^6.
Computes the fraction of theta-space where F_b(theta; R) > 0 for each
sample, yielding a posterior distribution over envelope volume fractions.
Reports mean and CI90/CI95.

Domains
-------
  full           : all of [-4.5, 4.5]^6
  core           : max|theta| <= 3.0
  expanded_shell : 3.0 < max|theta| <= 4.35
  extreme_shell  : max|theta| > 4.35

Inputs
------
results/seed_audit/seed_fdr_tiered.csv         — seed evaluation results
results/surrogates/gp_posterior_predictions.csv — fitted GP predictions

Outputs
-------
results/envelopes/envelope_volumes.csv   — per-(market, budget, domain) VF stats
results/envelopes/envelope_summary.md    — narrative envelope summary

Reference: Aziz (2026), "Through the Hidden States," Section 7 and Figure 4.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from scipy.stats.qmc import Sobol
    _HAS_SOBOL = True
except ImportError:
    _HAS_SOBOL = False

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

PROJECT_ROOT = Path(__file__).parent.parent
AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "seed_audit"
)
OUT_DIR = PROJECT_ROOT / "results" / "envelopes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_MIN, THETA_MAX = -4.5, 4.5
THETA_COLS = [f"theta_{j+1}" for j in range(6)]
MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BUDGETS = [5, 10]
MIN_NOISE_S = 1e-6
CANDIDATE_NOISE_FACTOR = 2.0
N_GRID = 4096    # Sobol points (power of 2 for Sobol)
N_SAMPLES = 1000 # independent pointwise posterior draws (no m×m covariance needed)
SEED = 42
DOMAINS = {
    "full":           lambda x: np.ones(len(x), dtype=bool),
    "core":           lambda x: np.max(np.abs(x), axis=1) <= 3.0,
    "expanded_shell": lambda x: (np.max(np.abs(x), axis=1) > 3.0)
                                & (np.max(np.abs(x), axis=1) <= 4.35),
    "extreme_shell":  lambda x: np.max(np.abs(x), axis=1) > 4.35,
}
SEP = "=" * 68


def normalize_theta(X_raw: np.ndarray) -> np.ndarray:
    return (X_raw - THETA_MIN) / (THETA_MAX - THETA_MIN)


def per_point_alpha(se: np.ndarray, tier: np.ndarray, y_std: float) -> np.ndarray:
    alpha_raw = np.where(np.isfinite(se) & (se > 0), se ** 2, 0.0)
    alpha_raw = np.where(tier == "candidate", alpha_raw * CANDIDATE_NOISE_FACTOR, alpha_raw)
    fallback = float(np.nanmedian(alpha_raw[alpha_raw > 0])) if np.any(alpha_raw > 0) else 0.01
    alpha_raw = np.where(alpha_raw <= 0, fallback, alpha_raw)
    return np.maximum(alpha_raw / max(y_std ** 2, 1e-12), MIN_NOISE_S)


def fit_gp(X_raw, y, se, tier, label=""):
    X = normalize_theta(X_raw)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    y_s = (y - y_mean) / y_std
    alpha_s = per_point_alpha(se, tier, y_std)
    kernel = ConstantKernel(1.0, (1e-3, 10.0)) * Matern(
        length_scale=np.ones(6), length_scale_bounds=(0.05, 5.0), nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha_s, normalize_y=False,
        n_restarts_optimizer=5, random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y_s)
    return gp, y_mean, y_std


def make_sobol_grid() -> np.ndarray:
    """Return an (N_GRID, 6) array of theta points covering [-4.5, 4.5]^6."""
    if _HAS_SOBOL:
        sampler = Sobol(d=6, scramble=True, seed=SEED)
        unit = sampler.random(n=N_GRID)
    else:
        rng = np.random.default_rng(SEED)
        unit = rng.uniform(0, 1, size=(N_GRID, 6))
    return THETA_MIN + unit * (THETA_MAX - THETA_MIN)


def envelope_stats(
    vol_fracs: np.ndarray, label: str
) -> dict:
    """Summarise a 1-D distribution of volume fractions."""
    return {
        "label": label,
        "mean": float(np.mean(vol_fracs)),
        "median": float(np.median(vol_fracs)),
        "std": float(np.std(vol_fracs)),
        "ci90_lo": float(np.percentile(vol_fracs, 5.0)),
        "ci90_hi": float(np.percentile(vol_fracs, 95.0)),
        "ci95_lo": float(np.percentile(vol_fracs, 2.5)),
        "ci95_hi": float(np.percentile(vol_fracs, 97.5)),
        "n_samples": len(vol_fracs),
    }


def main() -> None:
    print("Loading tiered FDR data...")
    tiered = pd.read_csv(AUDIT_DIR / "seed_fdr_tiered.csv")

    print("Generating Sobol grid...")
    X_grid = make_sobol_grid()
    X_grid_norm = normalize_theta(X_grid)
    domain_masks = {name: fn(X_grid) for name, fn in DOMAINS.items()}
    print(f"  Grid: {N_GRID} points in R^6")
    for name, mask in domain_masks.items():
        pct = 100.0 * mask.sum() / len(mask)
        print(f"  Domain '{name}': {mask.sum()} / {N_GRID} points ({pct:.1f}%)")

    result_rows: list[dict] = []

    print(f"\n{SEP}")
    print("ESTIMATING ENVELOPE VOLUMES")
    print(SEP)

    for mkt in MARKETS:
        for b in BUDGETS:
            sub = (tiered[(tiered["market"] == mkt) & (tiered["b"] == b)]
                   .copy().reset_index(drop=True))
            if sub.empty:
                continue

            X_raw = sub[THETA_COLS].to_numpy(float)
            y = sub["mean"].to_numpy(float)
            se = sub["se"].to_numpy(float)
            tier = sub["evidence_tier"].to_numpy(str)

            gp, y_mean, y_std = fit_gp(X_raw, y, se, tier, f"{mkt} b={b}")

            # Posterior mean and std at grid points (no observation noise)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu_s, sigma_s = gp.predict(X_grid_norm, return_std=True)
            mu = mu_s * y_std + y_mean
            sigma = sigma_s * y_std

            # Prob-positive from analytical GP posterior (no obs noise)
            prob_pos = norm.cdf(mu / np.maximum(sigma, 1e-9))

            # Independent pointwise posterior draws: draw Z_i ~ N(mu_i, sigma_i^2)
            # independently for each grid point. This avoids the O(m^2) full posterior
            # covariance while giving a valid lower bound on CI width (positive GP
            # correlations mean true CI is wider than this estimate).
            rng = np.random.default_rng(SEED)
            noise = rng.standard_normal(size=(len(mu), N_SAMPLES))
            # shape (N_GRID, N_SAMPLES): each column is one posterior draw
            y_samples = mu[:, None] + sigma[:, None] * noise

            label = f"{mkt}_b{b}"
            print(f"\n  {label}")

            for domain_name, mask in domain_masks.items():
                if mask.sum() == 0:
                    continue
                # Analytical prob-positive volume fraction
                analytic_vf = float(np.mean(prob_pos[mask] > 0.5))
                # Posterior sample volume fractions
                sample_vf = (y_samples[mask] > 0).mean(axis=0)  # (N_SAMPLES,)
                stats = envelope_stats(sample_vf, f"{label}_{domain_name}")
                n_domain = int(mask.sum())
                row = {
                    "market": mkt,
                    "budget": b,
                    "domain": domain_name,
                    "n_grid_points_in_domain": n_domain,
                    "analytic_vf_prob_pos_gt50pct": analytic_vf,
                    **stats,
                }
                result_rows.append(row)
                print(
                    f"    {domain_name:15s}: mean={stats['mean']:.4f}"
                    f"  CI90=[{stats['ci90_lo']:.4f},{stats['ci90_hi']:.4f}]"
                    f"  analytic={analytic_vf:.4f}"
                )

    df = pd.DataFrame(result_rows)
    out_csv = OUT_DIR / "envelope_volumes.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ------------------------------------------------------------------ #
    # Markdown summary
    # ------------------------------------------------------------------ #
    lines: list[str] = [
        "# Seed-Phase Envelope Volume Estimates",
        "",
        "## Method",
        "",
        f"- GP: ARD Matern-2.5, noise-aware (per-point alpha = SE^2 / y_std^2)",
        f"- Grid: {N_GRID} Sobol points in theta in [-4.5, 4.5]^6",
        f"- Posterior samples: N={N_SAMPLES} independent pointwise draws Z_i ~ N(mu_i, sigma_i^2)",
        "- Volume fraction: fraction of grid points where sample > 0 (per draw)",
        "- CI is a conservative lower bound: independent draws ignore positive GP correlations",
        "- Domains: full, core (max|theta|<=3), expanded_shell (3<max|theta|<=4.35),",
        "           extreme_shell (max|theta|>4.35)",
        "",
        "## Results",
        "",
    ]

    for mkt in MARKETS:
        lines.append(f"### {mkt}")
        lines.append("")
        lines.append("| Budget | Domain | Mean VF | CI90 | CI95 | Analytic (prob>0.5) |")
        lines.append("|--------|--------|---------|------|------|---------------------|")
        for b in BUDGETS:
            for domain in DOMAINS:
                sub = df[(df["market"] == mkt) & (df["budget"] == b) & (df["domain"] == domain)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                lines.append(
                    f"| b={b} | {domain} | {r['mean']:.4f} |"
                    f" [{r['ci90_lo']:.4f},{r['ci90_hi']:.4f}] |"
                    f" [{r['ci95_lo']:.4f},{r['ci95_hi']:.4f}] |"
                    f" {r['analytic_vf_prob_pos_gt50pct']:.4f} |"
                )
        lines.append("")

    # Cross-market comparison (full domain, b=5)
    lines.append("## Cross-Market Envelope at b=5 (full domain)")
    lines.append("")
    lines.append("| Market | Mean VF | CI90 |")
    lines.append("|--------|---------|------|")
    for mkt in MARKETS:
        sub = df[(df["market"] == mkt) & (df["budget"] == 5) & (df["domain"] == "full")]
        if not sub.empty:
            r = sub.iloc[0]
            lines.append(
                f"| {mkt} | {r['mean']:.4f} |"
                f" [{r['ci90_lo']:.4f},{r['ci90_hi']:.4f}] |"
            )
    lines.append("")

    # b=5 vs b=10 comparison (ETH full domain)
    lines.append("## ETH Envelope: b=5 vs b=10 (full domain)")
    lines.append("")
    for b in BUDGETS:
        sub = df[(df["market"] == "ETHUSDT") & (df["budget"] == b) & (df["domain"] == "full")]
        if not sub.empty:
            r = sub.iloc[0]
            lines.append(
                f"- b={b}: mean={r['mean']:.4f}"
                f"  CI90=[{r['ci90_lo']:.4f},{r['ci90_hi']:.4f}]"
            )
    lines.append("")
    lines.append("---")
    lines.append(
        "Note: volume fractions are relative to the sampled domain (uniform Sobol coverage)."
        " Not calibrated to absolute theta-space volume. Use for relative comparisons only."
    )

    out_md = OUT_DIR / "envelope_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_md}")

    print(f"\n{SEP}")
    print("DONE")
    print(f"  {out_csv}")
    print(f"  {out_md}")
    print(SEP)


if __name__ == "__main__":
    main()
