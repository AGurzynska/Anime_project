"""
Analyze anime portal Users CSV:
1) Describe columns (missing, uniques, summary stats).
2) Build a pipeline of how far users got with watching anime.

Usage:
    python analyze_anime_users.py path/to/users.csv

Outputs (printed + CSVs):
    - profile_missing.csv
    - profile_uniques.csv
    - profile_numeric_stats.csv
    - stage_distribution.csv
    - funnel_counts.csv
    - funnel_percentages.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC_INT_COLS = [
    "user_watching", "user_completed", "user_onhold", "user_dropped", "user_plantowatch", "stats_rewatched",
    "stats_episodes", "user_id"
]
NUMERIC_FLOAT_COLS = ["stats_mean_score", "user_days_spent_watching"]
DATE_COLS = ["birth_date", "join_date", "last_online"]
CAT_COLS = ["username", "gender", "location", "access_rank"]

def load_users_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV, parse dates, and coerce numerics safely."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN", "null", "None"])
    # Coerce numerics
    for c in NUMERIC_INT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in NUMERIC_FLOAT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    # Parse dates
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

# ---------- (1) DATA PROFILE ----------
def profile_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Create a CSV with data types and info on missing data for each column"""
    out = pd.DataFrame({
        "column": df.columns,
        "dtype": [df[c].dtype for c in df.columns],
        "non_null": df.notna().sum().values,
        "missing": df.isna().sum().values,
    })
    out["missing_pct"] = (out["missing"] / len(df)).round(4)
    return out.sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)

def profile_uniques(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DF with unique values for each column"""
    return pd.DataFrame({
        "column": df.columns,
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns]
    }).sort_values(["n_unique", "column"], ascending=[False, True], ignore_index=True)

def profile_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create statistics for columns with numerical data"""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return pd.DataFrame(columns=["column"])
    desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={
        "50%": "median", "25%": "p25", "75%": "p75"
    })
    # Add additional robust stats
    desc["missing"] = df[num_cols].isna().sum()
    desc["missing_pct"] = (desc["missing"] / len(df)).round(4)
    desc["zeros"] = (df[num_cols] == 0).sum()
    desc["nonzero_pct"] = (1 - (desc["zeros"] / desc["count"])).round(4)
    desc.index.name = "column"
    return desc.reset_index()

def print_quick_categoricals(df: pd.DataFrame, topk: int = 10) -> None:
    """Print info on categorical variables"""
    cat_cols = [c for c in df.columns if c in CAT_COLS and c in df]
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(topk)
        print(f"\nTop {topk} '{c}' values:")
        print(vc.to_string())

# ---------- (2) “HOW FAR USERS GOT IN WATCHING ANIME” PIPELINE ----------
def _gt0(s: pd.Series) -> pd.Series:
    return (s.fillna(0) > 0)

def compute_user_stage(df: pd.DataFrame) -> pd.Series:
    """
    Assign each user a single 'furthest progress stage'. Priority order:
    Completed ≥1 > Watching ≥1 > Started (On-Hold) > Started (Dropped) > Plan only > No activity.
    """
    has_completed = _gt0(df.get("user_completed", pd.Series(dtype="Int64")))
    has_watching  = _gt0(df.get("user_watching",  pd.Series(dtype="Int64")))
    has_onhold    = _gt0(df.get("user_onhold",    pd.Series(dtype="Int64")))
    has_dropped   = _gt0(df.get("user_dropped",   pd.Series(dtype="Int64")))
    has_ptw       = _gt0(df.get("user_plantowatch", pd.Series(dtype="Int64")))
    any_activity  = has_completed | has_watching | has_onhold | has_dropped | has_ptw

    stage = pd.Series("No list activity", index=df.index, dtype="string")
    stage = stage.mask(any_activity & has_ptw & ~(has_watching | has_completed | has_onhold | has_dropped), "Plan to Watch only")
    stage = stage.mask(any_activity & has_dropped & ~(has_watching | has_completed), "Started (Dropped)")
    stage = stage.mask(any_activity & has_onhold  & ~(has_watching | has_completed), "Started (On-Hold)")
    stage = stage.mask(has_watching & ~has_completed, "Watching ≥1")
    stage = stage.mask(has_completed, "Completed ≥1")

    # Categorical ordering for nice sorting
    stage = stage.astype(pd.CategoricalDtype(ordered=True, categories=[
        "Plan to Watch only",
        "Started (Dropped)",
        "Started (On-Hold)",
        "Watching ≥1",
        "Completed ≥1",
        "No list activity",
    ]))
    return stage

def compute_funnel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Funnel steps:
        Step 0: Any activity (any of the 5 list counters > 0)
        Step 1: Started (watching + completed + onhold + dropped > 0)
        Step 2: Currently watching (user_watching > 0)
        Step 3: Completed something (user_completed > 0)
    Side exits (not steps but context): Dropped > 0, On-hold > 0
    """
    has_ptw = _gt0(df.get("user_plantowatch", pd.Series(dtype="Int64")))
    started = _gt0(df.get("user_watching", pd.Series(dtype="Int64"))) \
            | _gt0(df.get("user_completed", pd.Series(dtype="Int64"))) \
            | _gt0(df.get("user_onhold", pd.Series(dtype="Int64"))) \
            | _gt0(df.get("user_dropped", pd.Series(dtype="Int64")))
    any_activity = started | has_ptw
    watching_now = _gt0(df.get("user_watching", pd.Series(dtype="Int64")))
    completed    = _gt0(df.get("user_completed", pd.Series(dtype="Int64")))
    dropped_any  = _gt0(df.get("user_dropped", pd.Series(dtype="Int64")))
    onhold_any   = _gt0(df.get("user_onhold", pd.Series(dtype="Int64")))

    counts = pd.DataFrame({
        "step": ["Any activity", "Started", "Watching now", "Completed ≥1", "Dropped ≥1 (side exit)", "On-Hold ≥1 (side exit)"],
        "count": [
            int(any_activity.sum()),
            int(started.sum()),
            int(watching_now.sum()),
            int(completed.sum()),
            int(dropped_any.sum()),
            int(onhold_any.sum()),
        ],
    })
    total_users = len(df)
    counts["pct_of_users"] = (counts["count"] / total_users * 100).round(2)
    # Percent of previous step for the core funnel (first 4 rows)
    prev = counts.loc[:3, "count"].shift(1)
    counts.loc[:3, "pct_of_prev_step"] = (counts.loc[:3, "count"] / prev * 100).round(2)
    counts.loc[counts.index[0], "pct_of_prev_step"] = np.nan  # N/A for first step

    # Percentage table only for the four core steps
    funnel_pct = counts.loc[:3, ["step", "pct_of_users", "pct_of_prev_step"]].reset_index(drop=True)
    return counts, funnel_pct

def analyze(csv_path: str | Path, out_dir: str | Path | None = None) -> None:
    out_dir = Path(out_dir or Path(csv_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_users_csv(csv_path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    # (1) Profile
    missing = profile_missing(df)
    uniques = profile_uniques(df)
    numeric = profile_numeric_stats(df)

    # Save profiles
    missing.to_csv(out_dir / "profile_missing.csv", index=False)
    uniques.to_csv(out_dir / "profile_uniques.csv", index=False)
    numeric.to_csv(out_dir / "profile_numeric_stats.csv", index=False)

    # Quick categorical peek in console
    print_quick_categoricals(df, topk=10)

    # (2) Stages + Funnel
    stage = compute_user_stage(df)
    stage_counts = stage.value_counts(dropna=False).rename_axis("stage").reset_index(name="count")
    stage_counts["pct_of_users"] = (stage_counts["count"] / len(df) * 100).round(2)
    stage_counts.to_csv(out_dir / "stage_distribution.csv", index=False)

    funnel_counts, funnel_pct = compute_funnel(df)
    funnel_counts.to_csv(out_dir / "funnel_counts.csv", index=False)
    funnel_pct.to_csv(out_dir / "funnel_percentages.csv", index=False)

    # Console summary
    print("\n=== Stage distribution ===")
    print(stage_counts.to_string(index=False))
    print("\n=== Funnel (counts) ===")
    print(funnel_counts.to_string(index=False))
    print("\n=== Funnel (percentages of users / previous step) ===")
    print(funnel_pct.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the users CSV.\nExample:\n  python analyze_anime_users.py data/users.csv")
        sys.exit(1)
    analyze(sys.argv[1])
