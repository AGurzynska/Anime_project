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
