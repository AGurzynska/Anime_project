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
