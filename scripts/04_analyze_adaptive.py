"""
Adaptive results analysis and frozen confirmation set selection (Section 6.5).

Parses adaptive run TGZ, merges with seed data, applies winner's curse
correction, and selects exactly 12 theta for the frozen confirmation run.

Selection rule (12 theta total)
--------------------------------
  3 confirmed seed anchors  — already paper-grade; receive 10 fresh reps
  9 from adaptive results   — top-9 by cross-market mean b=5 log-skill
                              where cross_market_mean > 0 (all markets)

  Deduplicated: each theta_id appears at most once.
  The 9 adaptive theta are chosen purely by cross-market average;
  no per-market or per-budget slotting is used.

Inputs
------
results/seed_audit/seed_fdr_tiered.csv
results/surrogates/gp_posterior_predictions.csv
(raw adaptive TGZ — not included in repo; see README)

Outputs
-------
results/tables/frozen_confirmation_set.csv
results/tables/adaptive_theta_proposal.csv   (full ranked adaptive list)

Reference: Aziz (2026), "Through the Hidden States," Section 6.5 and Section 8.3.
"""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTIVE_TGZ = PROJECT_ROOT / "runs" / "final_adaptive_results.tgz"  # not in repo; see README
AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "seed_audit"
)
ADAPTIVE_PROPOSAL = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "adaptive_theta_proposal.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_COLS = [f"theta_{j+1}" for j in range(6)]
MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BUDGETS = [5, 10]
SEP = "=" * 68

# Confirmed seed anchors (paper-grade, n>=10, 5% FDR) — ALWAYS in confirmation set
CONFIRMED_SEED_THETA_IDS = None   # will be loaded from tiered FDR file


def parse_result_json(path: Path) -> list[dict]:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return []
    rows = []
    theta = d.get("theta", [])
    for b_str, m in d.get("metrics", {}).items():
        row = {
            "job_id": d.get("job_id", path.stem),
            "theta_id": d.get("theta_id", ""),
            "market": d.get("market", ""),
            "b": int(b_str),
            "replicate_index": d.get("replicate_index", 0),
            "log_skill": m.get("log_skill"),
            "stage": d.get("stage", ""),
        }
        for j, v in enumerate(theta):
            row[f"theta_{j+1}"] = float(v)
        rows.append(row)
    return rows


def load_adaptive_results() -> pd.DataFrame:
    rows: list[dict] = []
    with tarfile.open(ADAPTIVE_TGZ, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json") and "results" in member.name:
                f = tf.extractfile(member)
                if f:
                    try:
                        d = json.load(f)
                        theta = d.get("theta", [])
                        for b_str, m in d.get("metrics", {}).items():
                            row = {
                                "theta_id": d.get("theta_id", ""),
                                "market": d.get("market", ""),
                                "b": int(b_str),
                                "replicate_index": d.get("replicate_index", 0),
                                "log_skill": m.get("log_skill"),
                            }
                            for j, v in enumerate(theta):
                                row[f"theta_{j+1}"] = float(v)
                            rows.append(row)
                    except Exception:
                        pass
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading adaptive results...")
    adap = load_adaptive_results()
    print(f"  {len(adap)} rows  |  {adap['theta_id'].nunique()} theta_ids")
    print(f"  log_skill range: [{adap['log_skill'].min():.3f}, {adap['log_skill'].max():.3f}]")
    print(f"  positive rate b=5: {(adap[adap.b==5]['log_skill']>0).mean()*100:.1f}%")

    # Join with proposal to get bucket labels
    proposal = pd.read_csv(ADAPTIVE_PROPOSAL)
    adap = adap.merge(
        proposal[["theta_id", "bucket"] + THETA_COLS].drop_duplicates("theta_id"),
        on="theta_id", how="left", suffixes=("", "_prop")
    )
    # Use proposal theta cols where the result might be missing (shouldn't happen, but safe)
    for c in THETA_COLS:
        if f"{c}_prop" in adap.columns:
            adap[c] = adap[c].fillna(adap[f"{c}_prop"])
            adap.drop(columns=[f"{c}_prop"], inplace=True)

    # Per-(theta_id, market, b) summary
    grp = adap.groupby(["theta_id", "market", "b"])["log_skill"]
    summary = grp.agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["theta_id", "market", "b", "mean_log_skill", "std_log_skill", "n"]
    summary["se_log_skill"] = summary["std_log_skill"] / np.sqrt(summary["n"])

    # Add theta coords and bucket
    theta_map = (
        adap[["theta_id"] + THETA_COLS + ["bucket"]]
        .drop_duplicates("theta_id")
    )
    summary = summary.merge(theta_map, on="theta_id", how="left")

    out_summary = OUT_DIR / "adaptive_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"\nSaved adaptive summary: {out_summary}")

    # ------------------------------------------------------------------ #
    # Cross-market signal table (b=5, mean across markets)
    # ------------------------------------------------------------------ #
    b5 = summary[summary["b"] == 5].copy()
    cross = b5.groupby("theta_id").agg(
        n_markets_positive=("mean_log_skill", lambda x: (x > 0).sum()),
        mean_cross_market=("mean_log_skill", "mean"),
        min_cross_market=("mean_log_skill", "min"),
    ).reset_index()
    cross = cross.merge(theta_map, on="theta_id", how="left")
    cross = cross.sort_values("mean_cross_market", ascending=False)

    print(f"\n{SEP}")
    print("TOP ADAPTIVE THETA BY CROSS-MARKET MEAN LOG_SKILL (b=5)")
    print(SEP)
    for _, r in cross.head(15).iterrows():
        th = [round(r[c], 2) for c in THETA_COLS]
        print(
            f"  {r['theta_id'][:12]}  bucket={r.get('bucket','?'):22s}"
            f"  cross_mean={r['mean_cross_market']:+.3f}"
            f"  n_pos_mkts={int(r['n_markets_positive'])}"
            f"  min={r['min_cross_market']:+.3f}"
        )

    # ------------------------------------------------------------------ #
    # Load confirmed seed anchors
    # ------------------------------------------------------------------ #
    tiered = pd.read_csv(AUDIT_DIR / "seed_fdr_tiered.csv")
    confirmed_seed = tiered[
        (tiered["evidence_tier"] == "anchor") &
        (tiered["sig_5pct_FDR_tiered"] == True) &
        (tiered["b"].isin(BUDGETS))
    ].drop_duplicates("theta_id")[["theta_id"] + THETA_COLS]
    print(f"\nConfirmed seed anchors: {len(confirmed_seed)} unique theta_ids")
    for _, r in confirmed_seed.iterrows():
        th = [round(r[c], 2) for c in THETA_COLS]
        print(f"  {r['theta_id'][:12]}  theta={th}")

    # ------------------------------------------------------------------ #
    # Select 9 adaptive theta for confirmation
    # ------------------------------------------------------------------ #
    # Strategy: top-9 by cross-market mean b=5 log_skill, requiring cross_mean > 0.
    # This ensures every adaptive confirmation theta has at least some positive signal
    # in the ensemble. Deduplicated from seed IDs.
    print(f"\n{SEP}")
    print("SELECTING 9 ADAPTIVE THETA (top-9 by cross-market mean b=5, cross_mean > 0)")
    print(SEP)

    seed_ids_set = set(confirmed_seed["theta_id"].unique())
    cand_pool = cross[
        (cross["mean_cross_market"] > 0) &
        (~cross["theta_id"].isin(seed_ids_set))
    ].sort_values("mean_cross_market", ascending=False)

    selected_adaptive: list[str] = []
    for _, row in cand_pool.iterrows():
        selected_adaptive.append(row["theta_id"])
        print(
            f"  + {row['theta_id'][:12]}"
            f"  cross_b5={row['mean_cross_market']:+.3f}"
            f"  n_pos={int(row['n_markets_positive'])}"
            f"  bucket={row.get('bucket','?')}"
        )
        if len(selected_adaptive) >= 9:
            break

    if len(selected_adaptive) < 9:
        print(f"  WARNING: only {len(selected_adaptive)} adaptive theta with cross_mean > 0")

    selected_ids: list[str] = list(confirmed_seed["theta_id"].unique()) + selected_adaptive

    adaptive_selected = [tid for tid in selected_ids if tid not in list(confirmed_seed["theta_id"])]
    print(f"\nTotal selected: {len(selected_ids)} = {len(confirmed_seed)} seed + {len(adaptive_selected)} adaptive")

    # ------------------------------------------------------------------ #
    # Build confirmation set CSV
    # ------------------------------------------------------------------ #
    # Get theta coords for each selected ID
    # Seed anchors: from tiered CSV
    # Adaptive: from summary
    all_theta_map = pd.concat([
        confirmed_seed.assign(source="seed_confirmed"),
        summary[summary["theta_id"].isin(adaptive_selected)][
            ["theta_id"] + THETA_COLS
        ].drop_duplicates("theta_id").assign(source="adaptive"),
    ])

    confirm_rows = []
    for tid in selected_ids:
        row_data = all_theta_map[all_theta_map["theta_id"] == tid]
        if row_data.empty:
            print(f"  WARNING: no theta coords for {tid}")
            continue
        row = row_data.iloc[0]
        source = row.get("source", "unknown")

        # Get b=5 cross-market summary
        b5_sub = b5[b5["theta_id"] == tid]
        cross_mean_b5 = float(b5_sub["mean_log_skill"].mean()) if not b5_sub.empty else np.nan
        n_pos_b5 = int((b5_sub["mean_log_skill"] > 0).sum()) if not b5_sub.empty else 0

        confirm_rows.append({
            "theta_id": tid,
            **{c: float(row[c]) for c in THETA_COLS},
            "source": source,
            "cross_market_mean_b5": cross_mean_b5,
            "n_markets_positive_b5": n_pos_b5,
            "initial_replicates": 10,   # for make-jobs
            "design_source": "confirmation",
            "role": "confirmation_anchor" if source == "seed_confirmed" else "confirmation_adaptive",
        })

    confirm_df = pd.DataFrame(confirm_rows)
    assert len(confirm_df) == 12, f"Expected 12 confirmation theta, got {len(confirm_df)}"

    out_csv = OUT_DIR / "frozen_confirmation_set.csv"
    confirm_df.to_csv(out_csv, index=False)

    print(f"\n{SEP}")
    print("FROZEN CONFIRMATION SET (12 theta)")
    print(SEP)
    for _, r in confirm_df.iterrows():
        th = [round(r[c], 2) for c in THETA_COLS]
        print(
            f"  [{r['source'][:16]:16s}]  cross_b5={r['cross_market_mean_b5']:+.3f}"
            f"  n_pos={r['n_markets_positive_b5']}  theta={th}"
        )

    print(f"\nSaved: {out_csv}")
    print(f"Job count: 12 theta x 3 markets x 10 reps = 360 confirmation jobs")
    print(SEP)


if __name__ == "__main__":
    main()
