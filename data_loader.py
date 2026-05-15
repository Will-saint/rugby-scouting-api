"""CSV data loader with in-memory cache (5-min TTL)."""
import os
import time
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# Production : données bundlées dans api/data/
# Dev fallback : lire depuis app1 si api/data/ absent
_LOCAL_DATA = Path(__file__).parent / "data" / "seasons"
_APP1_DATA  = Path(__file__).parent.parent.parent / "app1-rating-engine" / "rugby-rating-engine" / "data" / "seasons"
SEASONS_DIR = _LOCAL_DATA if _LOCAL_DATA.exists() else _APP1_DATA
DATA_DIR    = SEASONS_DIR.parent
AVAILABLE_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
DEFAULT_SEASON = "2025-2026"

TIER_ORDER = {"LEGENDAIRE": 5, "OR": 4, "ARGENT": 3, "BRONZE": 2, "STANDARD": 1}

_cache: dict[str, tuple[pd.DataFrame, float]] = {}
CACHE_TTL = 300  # 5 minutes


def _load_season(season: str) -> pd.DataFrame:
    now = time.time()
    if season in _cache:
        df, ts = _cache[season]
        if now - ts < CACHE_TTL:
            return df

    path = SEASONS_DIR / season / "players_scored.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    # Keep numeric columns as numeric — do not replace NaN with None at load time
    # (that would coerce numeric cols to object dtype and break .mean() etc.)
    _cache[season] = (df, now)
    return df


def get_df(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    if season not in AVAILABLE_SEASONS:
        season = DEFAULT_SEASON
    return _load_season(season)


def get_seasons() -> list[str]:
    return [s for s in AVAILABLE_SEASONS if (SEASONS_DIR / s / "players_scored.csv").exists()]


def safe_float(v) -> Optional[float]:
    try:
        f = float(v) if v is not None else None
        if f is None or (f != f) or f == float("inf") or f == float("-inf"):
            return None
        return f
    except (TypeError, ValueError):
        return None


def safe_int(v) -> Optional[int]:
    try:
        f = float(v) if v is not None else None
        if f is None or (f != f) or f == float("inf") or f == float("-inf"):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def safe_str(v) -> Optional[str]:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s and s != "nan" else None


def rating_to_tier(r: Optional[float]) -> str:
    if r is None:
        return "STANDARD"
    if r >= 90:
        return "LEGENDAIRE"
    if r >= 84:
        return "OR"
    if r >= 77:
        return "ARGENT"
    if r >= 70:
        return "BRONZE"
    return "STANDARD"


def row_to_summary(row: pd.Series) -> dict:
    rating = safe_float(row.get("rating")) or safe_float(row.get("display_rating")) or 40.0
    return {
        "lnr_slug":        safe_str(row.get("lnr_slug")),
        "name":            safe_str(row.get("name")),
        "team":            safe_str(row.get("team")),
        "position_group":  safe_str(row.get("position_group")),
        "position_label":  safe_str(row.get("position_label")) or safe_str(row.get("position_group")),
        "rating":          round(rating, 1),
        "tier":            rating_to_tier(rating),
        "age":             safe_int(row.get("age")),
        "height_cm":       safe_float(row.get("height_cm")),
        "weight_kg":       safe_float(row.get("weight_kg")),
        "nationality":     safe_str(row.get("nationality")),
        "photo_url":       safe_str(row.get("photo_url")),
        "confidence_badge": safe_str(row.get("confidence_badge")) or "Basse",
    }


def row_to_detail(row: pd.Series, history: list[dict]) -> dict:
    base = row_to_summary(row)
    base.update({
        "rating_raw": safe_float(row.get("rating_raw")),
        "age_factor": safe_float(row.get("age_factor")) or 0.0,
        "intl_bonus": safe_float(row.get("intl_bonus")) or 0.0,
        "form_score": safe_float(row.get("form_score")),
        "form_score_10": safe_float(row.get("form_score_10")),
        "form_trend": row.get("form_trend") or "→",
        "form_trend_10": row.get("form_trend_10") or "→",
        "minutes_played": safe_int(row.get("minutes_played")),
        "axis_att": safe_float(row.get("axis_att")),
        "axis_def": safe_float(row.get("axis_def")),
        "axis_ctrl": safe_float(row.get("axis_ctrl")),
        "axis_kick": safe_float(row.get("axis_kick")),
        "axis_pow": safe_float(row.get("axis_pow")),
        "axis_gabarit": safe_float(row.get("axis_gabarit")),
        "axis_disc": safe_float(row.get("axis_disc")),
        "tackles_per80": safe_float(row.get("tackles_per80")),
        "offloads_per80": safe_float(row.get("offloads_per80")),
        "line_breaks_per80": safe_float(row.get("line_breaks_per80")),
        "turnovers_won_per80": safe_float(row.get("turnovers_won_per80")),
        "tries_per80": safe_float(row.get("tries_per80")),
        "kick_points_per80": safe_float(row.get("kick_points_per80")),
        "yellow_cards": safe_int(row.get("yellow_cards")),
        "orange_cards": safe_int(row.get("orange_cards")),
        "red_cards": safe_int(row.get("red_cards")),
        "rating_intl": safe_float(row.get("rating_intl")),
        "team_intl": row.get("team_intl"),
        "matches_intl": safe_int(row.get("matches_intl")),
        "axis_course_intl": safe_float(row.get("axis_course_intl")),
        "axis_distrib_intl": safe_float(row.get("axis_distrib_intl")),
        "axis_kicking_intl": safe_float(row.get("axis_kicking_intl")),
        "axis_physique_intl": safe_float(row.get("axis_physique_intl")),
        "axis_rigueur_intl": safe_float(row.get("axis_rigueur_intl")),
        "axis_danger_intl": safe_float(row.get("axis_danger_intl")),
        "axis_melee_intl": safe_float(row.get("axis_melee_intl")),
        "history": history,
    })
    return base
