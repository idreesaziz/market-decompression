"""
GP-guided adaptive theta selection (Section 6.4 / Section 8.2).

Selects exactly 40 adaptive theta points from a novel Sobol pool filtered
to exclude points within 0.5 (Euclidean) of any seed theta.

Bucket allocation
-----------------
  SOL_b5               : 12  — SOL b=5 positive / high-prob region
  BTC_b5               : 10  — BTC b=5 positive / boundary region
  ETH_b5b10            :  8  — ETH b=5 (4) + b=10 (4) positive region
  shared_neighborhood  :  4  — local neighbourhood of confirmed BTC+ETH anchor
  ETH_extreme_nbhd     :  3  — local neighbourhood of confirmed ETH extreme anchor
  IMSPE_global         :  3  — maximum-uncertainty coverage
                           --
  TOTAL                : 40

Strategy: SOL/BTC/ETH buckets ranked by prob_positive from the relevant GP.
  Neighborhood buckets use small perturbations ranked by market-average GP
  posterior mean. IMSPE bucket maximises mean posterior_std across all 6 GPs.
  Novel pool: 16384 Sobol points, min distance 0.5 from all seed theta.

Inputs
------
results/seed_audit/seed_fdr_tiered.csv
results/surrogates/gp_posterior_predictions.csv

Output
------
results/tables/adaptive_theta_proposal.csv

Reference: Aziz (2026), "Through the Hidden States," Section 6.4 and Section 8.2.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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
OUT_DIR = PROJECT_ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_MIN, THETA_MAX = -4.5, 4.5
THETA_COLS = [f"theta_{j+1}" for j in range(6)]
MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BUDGETS = [5, 10]
MIN_NOISE_S = 1e-6
CANDIDATE_NOISE_FACTOR = 2.0
SEED = 42
N_POOL = 16384        # novel Sobol pool size
MIN_DIST_FROM_SEED = 0.5   # minimum Euclidean distance from any existing seed theta
MIN_DIST_BETWEEN_SELECTED = 0.3  # dedup: min distance between selected adaptive theta
SEP = "=" * 68

# Confirmed anchors for neighbourhood buckets
ANCHOR_SHARED = np.array([1.0, 0.5, -1.0, 0.5, -0.5, 1.0])    # BTC + ETH confirmed
ANCHOR_ETH_EXTREME = np.array([4.2, -4.2, 4.2, -4.2, 4.2, -4.2])  # ETH extreme confirmed

BUCKET_DEFS = [
    dict(name="SOL_b5",              n=12, market="SOLUSDT", budget=5),
    dict(name="BTC_b5",              n=10, market="BTCUSDT", budget=5),
    dict(name="ETH_b5",              n=4,  market="ETHUSDT", budget=5),
    dict(name="ETH_b10",             n=4,  market="ETHUSDT", budget=10),
    dict(name="shared_neighborhood", n=4,  market="shared",  budget=5),
    dict(name="ETH_extreme_nbhd",    n=3,  market="ETHUSDT", budget=5),
    dict(name="IMSPE_global",        n=3,  market="all",     budget="5,10"),
]


# ------------------------------------------------------------------ #
# GP helpers (duplicated from surrogate script for standalone running)
# ------------------------------------------------------------------ #

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


def predict_at(gp, y_mean, y_std, X_raw):
    X = normalize_theta(X_raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_s, sig_s = gp.predict(X, return_std=True)
    return mu_s * y_std + y_mean, sig_s * y_std


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def theta_id(theta: np.ndarray) -> str:
    key = json.dumps([round(float(v), 10) for v in theta], separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def make_sobol_pool(n: int = N_POOL, seed: int = SEED) -> np.ndarray:
    if _HAS_SOBOL:
        sampler = Sobol(d=6, scramble=True, seed=seed)
        unit = sampler.random(n=n)
    else:
        rng = np.random.default_rng(seed)
        unit = rng.uniform(0, 1, size=(n, 6))
    return THETA_MIN + unit * (THETA_MAX - THETA_MIN)


def exclude_near_existing(pool: np.ndarray, seed_theta: np.ndarray,
                          min_dist: float = MIN_DIST_FROM_SEED) -> np.ndarray:
    """Return boolean mask: True for pool points far enough from all seed theta."""
    dists = cdist(pool, seed_theta, metric="euclidean")
    return dists.min(axis=1) >= min_dist


def make_anchor_neighbourhood(anchor: np.ndarray, n_gen: int = 120,
                              seed: int = SEED) -> np.ndarray:
    """Generate perturbations of anchor, clipped to [-4.5, 4.5]."""
    rng = np.random.default_rng(seed)
    # Random perturbations: magnitude 0.2-0.9 per active dim
    deltas = rng.uniform(-0.9, 0.9, size=(n_gen, 6))
    # Randomise which dims are perturbed (on average 2-3 dims)
    active = rng.uniform(0, 1, size=(n_gen, 6)) < 0.45
    deltas = deltas * active
    # Vary perturbation scale
    scale = rng.uniform(0.3, 1.0, size=(n_gen, 1))
    candidates = anchor + deltas * scale
    return np.clip(candidates, THETA_MIN, THETA_MAX)


def greedy_diverse_select(
    pool: np.ndarray,
    scores: np.ndarray,
    n: int,
    already_selected: np.ndarray | None,
    min_dist: float = MIN_DIST_BETWEEN_SELECTED,
) -> np.ndarray:
    """
    Greedy diverse selection: pick top-score points while ensuring minimum
    pairwise distance from already-selected AND among newly selected.
    Returns indices into pool.
    """
    order = np.argsort(-scores)  # high score first
    selected_indices = []
    selected_pts = [] if already_selected is None or len(already_selected) == 0 else list(already_selected)

    for idx in order:
        if len(selected_indices) >= n:
            break
        pt = pool[idx]
        too_close = False
        for ref in selected_pts:
            if np.linalg.norm(pt - ref) < min_dist:
                too_close = True
                break
        if not too_close:
            selected_indices.append(idx)
            selected_pts.append(pt)

    return np.array(selected_indices, dtype=int)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    print("Loading tiered FDR data...")
    tiered = pd.read_csv(AUDIT_DIR / "seed_fdr_tiered.csv")

    # Seed theta matrix for exclusion
    seed_sub = tiered[tiered["b"] == 5][THETA_COLS].drop_duplicates()
    seed_theta = seed_sub.to_numpy(float)
    print(f"  Seed theta points: {len(seed_theta)}")

    # Fit all 6 GPs
    print(f"\n{SEP}\nFITTING 6 SURROGATES\n{SEP}")
    gps: dict[tuple, tuple] = {}
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
            gps[(mkt, b)] = (gp, y_mean, y_std)
            print(f"  {mkt} b={b}: fitted")

    # Novel Sobol pool
    print(f"\n{SEP}\nGENERATING NOVEL THETA POOL\n{SEP}")
    raw_pool = make_sobol_pool()
    novel_mask = exclude_near_existing(raw_pool, seed_theta)
    pool = raw_pool[novel_mask]
    print(f"  Raw pool: {len(raw_pool)}, after exclusion: {len(pool)}")

    # Pre-compute prob_positive and posterior_std at pool for all GPs
    print("\n  Evaluating surrogates at pool...")
    pool_pp: dict[tuple, np.ndarray] = {}  # (mkt, b) -> prob_positive at pool
    pool_std: dict[tuple, np.ndarray] = {} # (mkt, b) -> posterior_std at pool
    for (mkt, b), (gp, y_mean, y_std) in gps.items():
        mu, sigma = predict_at(gp, y_mean, y_std, pool)
        pool_pp[(mkt, b)] = norm.cdf(mu / np.maximum(sigma, 1e-9))
        pool_std[(mkt, b)] = sigma
        print(f"    {mkt} b={b}: mean(prob_positive)={pool_pp[(mkt,b)].mean():.3f}")

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}\nSELECTING 40 ADAPTIVE THETA POINTS\n{SEP}")

    selected_rows: list[dict] = []
    all_selected_pts: list[np.ndarray] = []   # for dedup across buckets

    def _add_selection(
        theta_pts: np.ndarray,
        bucket_name: str,
        acq_scores: np.ndarray,
        nearest_seed_theta: np.ndarray,
        nearest_seed_dist: np.ndarray,
        primary_mu: np.ndarray,
        primary_pp: np.ndarray,
        primary_std: np.ndarray,
        target_markets: str,
        target_budgets: str,
        proposed_reps: int = 2,
    ):
        for rank_in_bucket, i in enumerate(range(len(theta_pts))):
            th = theta_pts[i]
            all_selected_pts.append(th)
            selected_rows.append({
                "theta_id": theta_id(th),
                **{f"theta_{j+1}": float(th[j]) for j in range(6)},
                "bucket": bucket_name,
                "selection_rank_in_bucket": rank_in_bucket + 1,
                "acquisition_score": float(acq_scores[i]),
                "posterior_mean_primary": float(primary_mu[i]),
                "prob_positive_primary": float(primary_pp[i]),
                "posterior_std_primary": float(primary_std[i]),
                "nearest_seed_theta_id": str(nearest_seed_theta[i]),
                "nearest_seed_distance_L2": float(nearest_seed_dist[i]),
                "target_markets": target_markets,
                "target_budgets": target_budgets,
                "proposed_reps": proposed_reps,
            })

    def _nearest_seed_info(pts: np.ndarray):
        dists = cdist(pts, seed_theta)
        idx = dists.argmin(axis=1)
        # Just store the index as placeholder (no theta_id in seed_theta matrix)
        return np.array([f"seed_{j}" for j in idx]), dists.min(axis=1)

    # ---- Bucket 1: SOL b=5 — 12 theta, pure positive ----
    mkt, b = "SOLUSDT", 5
    pp = pool_pp[(mkt, b)]
    scores = pp  # rank by prob_positive
    idxs = greedy_diverse_select(pool, scores, 12, np.array(all_selected_pts) if all_selected_pts else None)
    th_pts = pool[idxs]
    mu_p, sig_p = predict_at(*gps[(mkt, b)], th_pts)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "SOL_b5", scores[idxs],
        ns_id, ns_dist, mu_p, pp[idxs], sig_p,
        "SOLUSDT,BTCUSDT,ETHUSDT", "5,10",
    )
    print(f"  SOL_b5: selected {len(idxs)}, mean(pp)={pp[idxs].mean():.3f}")

    # ---- Bucket 2: BTC b=5 — 10 theta (6 high-pp + 4 boundary) ----
    mkt, b = "BTCUSDT", 5
    pp = pool_pp[(mkt, b)]
    already = np.array(all_selected_pts)

    # 6 high prob_positive
    scores_high = pp.copy()
    idxs_high = greedy_diverse_select(pool, scores_high, 6, already)

    # 4 boundary: 0.4 < prob_positive < 0.75
    boundary_score = np.where((pp > 0.4) & (pp < 0.75), pp, 0.0)
    combined_already = np.vstack([already, pool[idxs_high]]) if len(already) > 0 else pool[idxs_high]
    idxs_boundary = greedy_diverse_select(pool, boundary_score, 4, combined_already)

    idxs_btc = np.concatenate([idxs_high, idxs_boundary])
    th_pts = pool[idxs_btc]
    mu_p, sig_p = predict_at(*gps[(mkt, b)], th_pts)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    acq = np.concatenate([scores_high[idxs_high], boundary_score[idxs_boundary]])
    _add_selection(
        th_pts, "BTC_b5", acq,
        ns_id, ns_dist, mu_p,
        np.concatenate([pp[idxs_high], pp[idxs_boundary]]),
        sig_p, "BTCUSDT,ETHUSDT,SOLUSDT", "5,10",
    )
    print(f"  BTC_b5: selected {len(idxs_btc)} (6 high + 4 boundary)")

    # ---- Bucket 3: ETH b=5 — 4 theta ----
    mkt, b = "ETHUSDT", 5
    pp = pool_pp[(mkt, b)]
    already = np.array(all_selected_pts)
    idxs = greedy_diverse_select(pool, pp, 4, already)
    th_pts = pool[idxs]
    mu_p, sig_p = predict_at(*gps[(mkt, b)], th_pts)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "ETH_b5", pp[idxs],
        ns_id, ns_dist, mu_p, pp[idxs], sig_p,
        "ETHUSDT,BTCUSDT,SOLUSDT", "5,10",
    )
    print(f"  ETH_b5: selected {len(idxs)}")

    # ---- Bucket 4: ETH b=10 — 4 theta ----
    mkt, b = "ETHUSDT", 10
    pp = pool_pp[(mkt, b)]
    already = np.array(all_selected_pts)
    idxs = greedy_diverse_select(pool, pp, 4, already)
    th_pts = pool[idxs]
    mu_p, sig_p = predict_at(*gps[(mkt, b)], th_pts)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "ETH_b10", pp[idxs],
        ns_id, ns_dist, mu_p, pp[idxs], sig_p,
        "ETHUSDT,BTCUSDT,SOLUSDT", "5,10",
    )
    print(f"  ETH_b10: selected {len(idxs)}")

    # ---- Bucket 5: shared_neighborhood — 4 theta near BTC+ETH confirmed anchor ----
    nbhd_raw = make_anchor_neighbourhood(ANCHOR_SHARED, n_gen=200, seed=SEED)
    nbhd_novel_mask = exclude_near_existing(nbhd_raw, seed_theta, min_dist=0.4)
    nbhd = nbhd_raw[nbhd_novel_mask]
    if len(nbhd) < 4:
        nbhd = nbhd_raw  # relax if too few survive
        print("  WARNING: shared_neighborhood: relaxed exclusion filter")

    pp_btc5 = norm.cdf(
        predict_at(*gps[("BTCUSDT", 5)], nbhd)[0]
        / np.maximum(predict_at(*gps[("BTCUSDT", 5)], nbhd)[1], 1e-9)
    )
    pp_eth5 = norm.cdf(
        predict_at(*gps[("ETHUSDT", 5)], nbhd)[0]
        / np.maximum(predict_at(*gps[("ETHUSDT", 5)], nbhd)[1], 1e-9)
    )
    mu_btc, sig_btc = predict_at(*gps[("BTCUSDT", 5)], nbhd)
    mu_eth, sig_eth = predict_at(*gps[("ETHUSDT", 5)], nbhd)
    # Acquisition: average posterior_mean across BTC and ETH
    nbhd_scores = (mu_btc + mu_eth) / 2.0
    already = np.array(all_selected_pts)
    nbhd_idxs = greedy_diverse_select(nbhd, nbhd_scores, 4, already, min_dist=0.3)
    th_pts = nbhd[nbhd_idxs]
    mu_primary = nbhd_scores[nbhd_idxs]
    pp_primary = ((pp_btc5[nbhd_idxs] + pp_eth5[nbhd_idxs]) / 2.0)
    sig_primary = ((sig_btc[nbhd_idxs] + sig_eth[nbhd_idxs]) / 2.0)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "shared_neighborhood", nbhd_scores[nbhd_idxs],
        ns_id, ns_dist, mu_primary, pp_primary, sig_primary,
        "BTCUSDT,ETHUSDT,SOLUSDT", "5,10",
    )
    print(f"  shared_neighborhood: selected {len(nbhd_idxs)}")

    # ---- Bucket 6: ETH_extreme_nbhd — 3 theta near ETH extreme anchor ----
    ext_raw = make_anchor_neighbourhood(ANCHOR_ETH_EXTREME, n_gen=200, seed=SEED + 1)
    ext_novel_mask = exclude_near_existing(ext_raw, seed_theta, min_dist=0.4)
    ext_nbhd = ext_raw[ext_novel_mask]
    if len(ext_nbhd) < 3:
        ext_nbhd = ext_raw
        print("  WARNING: ETH_extreme_nbhd: relaxed exclusion filter")

    mu_eth_ext, sig_eth_ext = predict_at(*gps[("ETHUSDT", 5)], ext_nbhd)
    pp_eth_ext = norm.cdf(mu_eth_ext / np.maximum(sig_eth_ext, 1e-9))
    ext_scores = mu_eth_ext
    already = np.array(all_selected_pts)
    ext_idxs = greedy_diverse_select(ext_nbhd, ext_scores, 3, already, min_dist=0.3)
    th_pts = ext_nbhd[ext_idxs]
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "ETH_extreme_nbhd", ext_scores[ext_idxs],
        ns_id, ns_dist,
        mu_eth_ext[ext_idxs], pp_eth_ext[ext_idxs], sig_eth_ext[ext_idxs],
        "ETHUSDT,BTCUSDT,SOLUSDT", "5,10",
    )
    print(f"  ETH_extreme_nbhd: selected {len(ext_idxs)}")

    # ---- Bucket 7: IMSPE_global — 3 theta, max avg posterior_std ----
    imspe_std = np.mean(
        np.stack([pool_std[(mkt, b)] for (mkt, b) in pool_std], axis=0),
        axis=0,
    )
    already = np.array(all_selected_pts)
    imspe_idxs = greedy_diverse_select(pool, imspe_std, 3, already)
    th_pts = pool[imspe_idxs]
    mu_imspe = np.zeros(3)  # pooled across GPs
    pp_imspe = np.zeros(3)
    sig_imspe = imspe_std[imspe_idxs]
    for (mkt, b), (gp, y_mean, y_std) in gps.items():
        mu_g, _ = predict_at(gp, y_mean, y_std, th_pts)
        pp_g = norm.cdf(mu_g / np.maximum(imspe_std[imspe_idxs], 1e-9))
        mu_imspe += mu_g
        pp_imspe += pp_g
    mu_imspe /= len(gps)
    pp_imspe /= len(gps)
    ns_id, ns_dist = _nearest_seed_info(th_pts)
    _add_selection(
        th_pts, "IMSPE_global", sig_imspe,
        ns_id, ns_dist, mu_imspe, pp_imspe, sig_imspe,
        "BTCUSDT,ETHUSDT,SOLUSDT", "5,10",
    )
    print(f"  IMSPE_global: selected {len(imspe_idxs)}")

    # ------------------------------------------------------------------ #
    # Finalise and save
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(selected_rows)
    assert len(df) == 40, f"Expected 40 adaptive theta, got {len(df)}"

    # Sanity: check uniqueness of theta_id
    assert df["theta_id"].nunique() == len(df), \
        "Duplicate theta_ids in adaptive proposal — hash collision or repeated point"

    out_csv = OUT_DIR / "adaptive_theta_proposal.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{SEP}")
    print("ADAPTIVE THETA PROPOSAL SUMMARY")
    print(SEP)
    print(f"  Total: {len(df)} unique theta points")
    for bucket_name in [d["name"] for d in BUCKET_DEFS]:
        sub = df[df["bucket"] == bucket_name]
        if sub.empty:
            continue
        print(
            f"  {bucket_name:25s}: n={len(sub):2d}"
            f"  mean(prob_pos)={sub['prob_positive_primary'].mean():.3f}"
            f"  mean(dist_to_seed)={sub['nearest_seed_distance_L2'].mean():.2f}"
        )

    print(f"\n  Job count if each theta x 3 markets x 2 reps: {len(df) * 3 * 2}")
    print(f"\nSaved: {out_csv}")
    print("\nDO NOT launch adaptive jobs until this proposal has been reviewed.")
    print(SEP)


if __name__ == "__main__":
    main()
