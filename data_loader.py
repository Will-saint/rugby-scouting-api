"""CSV data loader with in-memory cache (5-min TTL)."""
import json
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


# ── Awards ────────────────────────────────────────────────────────────────────

_awards_cache: Optional[dict] = None

def _load_awards() -> dict:
    global _awards_cache
    if _awards_cache is not None:
        return _awards_cache
    awards_path = DATA_DIR / "awards.json"
    if not awards_path.exists():
        _awards_cache = {}
        return _awards_cache
    with open(awards_path, encoding="utf-8") as f:
        _awards_cache = json.load(f)
    return _awards_cache


def get_player_badges(lnr_slug: str, team: str, season: str) -> list[dict]:
    """Return badge list for a player based on individual awards + team championships."""
    awards = _load_awards()
    if not awards:
        return []

    badges: list[dict] = []
    defs = awards.get("badge_definitions", {})

    # Individual awards
    for entry in awards.get("individual_awards", []):
        if entry.get("lnr_slug") == lnr_slug:
            for aw in entry.get("awards", []):
                badge_def = defs.get(aw["id"], {})
                badges.append({
                    "id": aw["id"],
                    "label": aw.get("label", badge_def.get("label", "")),
                    "short": badge_def.get("short", aw.get("label", "")),
                    "year": aw.get("year"),
                    "icon": badge_def.get("icon", "star"),
                    "color": badge_def.get("color", "#888"),
                })

    # Team championships — normalise team name for lookup
    team_norm = team or ""
    club_data = awards.get("club_championships", {})
    for club_key, titles in club_data.items():
        if club_key.lower() in team_norm.lower() or team_norm.lower() in club_key.lower():
            for bouclier_season in titles.get("bouclier", []):
                if bouclier_season == season:
                    def_b = defs.get("bouclier", {})
                    badges.append({
                        "id": "bouclier",
                        "label": f"Bouclier {bouclier_season[:4]}",
                        "short": "Bouclier",
                        "year": int(bouclier_season[:4]),
                        "icon": def_b.get("icon", "shield"),
                        "color": def_b.get("color", "#b94f3a"),
                    })
            for cc_season in titles.get("champions_cup", []):
                if cc_season == season:
                    def_cc = defs.get("champions_cup", {})
                    badges.append({
                        "id": "champions_cup",
                        "label": f"Champions Cup {cc_season[:4]}",
                        "short": "CC",
                        "year": int(cc_season[:4]),
                        "icon": def_cc.get("icon", "cup"),
                        "color": def_cc.get("color", "#7c3aed"),
                    })

    return badges

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
    lnr_slug = safe_str(row.get("lnr_slug")) or ""
    team = safe_str(row.get("team")) or ""
    season = safe_str(row.get("season")) or DEFAULT_SEASON
    return {
        "lnr_slug":        lnr_slug,
        "name":            safe_str(row.get("name")),
        "team":            team,
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
        "rating_intl":     safe_float(row.get("rating_intl")),
        "badges":          get_player_badges(lnr_slug, team, season),
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
        # Raw Naim intl stats (per-game averages from ESPN)
        "meters_run_intl": safe_float(row.get("meters_run_intl")),
        "clean_breaks_intl": safe_float(row.get("clean_breaks_intl")),
        "defenders_beaten_intl": safe_float(row.get("defenders_beaten_intl")),
        "passes_intl": safe_float(row.get("passes_intl")),
        "runs_intl": safe_float(row.get("runs_intl")),
        "lineouts_won_intl": safe_float(row.get("lineouts_won_intl")),
        "missed_tackles_intl": safe_float(row.get("missed_tackles_intl")),
        "tackles_intl": safe_float(row.get("tackles_intl")),
        "turnovers_conceded_intl": safe_float(row.get("turnovers_conceded_intl")),
        "penalties_conceded_intl": safe_float(row.get("penalties_conceded_intl")),
        "offloads_intl": safe_float(row.get("offloads_intl")),
        # Awards / badges
        "badges": get_player_badges(
            safe_str(row.get("lnr_slug")) or "",
            safe_str(row.get("team")) or "",
            safe_str(row.get("season")) or DEFAULT_SEASON,
        ),
        "history": history,
    })
    return base
