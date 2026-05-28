"""Match prediction — reuses the logic from the Streamlit engine."""
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

AXIS_COLS = ["axis_att", "axis_def", "axis_ctrl", "axis_kick", "axis_pow", "axis_gabarit"]


def _team_profile(df: pd.DataFrame, team: str) -> dict:
    t = df[df["team"] == team]
    if t.empty:
        return {}
    axes = {col: float(t[col].mean()) for col in AXIS_COLS if col in t.columns}
    avg_rating = float(t["rating"].mean()) if "rating" in t.columns else 60.0
    return {"avg_rating": round(avg_rating, 1), "axes": axes, "n_players": len(t)}


def predict_match(df: pd.DataFrame, home: str, away: str) -> dict:
    hp = _team_profile(df, home)
    ap = _team_profile(df, away)

    if not hp or not ap:
        missing = home if not hp else away
        return {"error": f"Team not found: {missing}"}

    hr = hp["avg_rating"]
    ar = ap["avg_rating"]
    diff = hr - ar

    # Logistic sigmoid — same curve as Streamlit predictor
    home_advantage = 3.5
    adj = diff + home_advantage
    home_win = float(1 / (1 + np.exp(-adj / 12)))
    away_win = float(1 / (1 + np.exp(adj / 12)))
    draw = max(0.0, 1.0 - home_win - away_win)

    # Normalize
    total = home_win + away_win + draw
    home_win /= total
    away_win /= total
    draw /= total

    # Estimated score (rudimentary)
    base_pts = 22
    score_home = round(base_pts + diff * 0.4 + home_advantage * 0.5)
    score_away = round(base_pts - diff * 0.4)

    # Key matchups
    matchups = []
    for axis, label in [("axis_att", "Attaque"), ("axis_def", "Défense"), ("axis_ctrl", "Contrôle")]:
        ha = hp["axes"].get(axis, 50)
        aa = ap["axes"].get(axis, 50)
        matchups.append({
            "axis": label,
            "home_val": round(ha, 1),
            "away_val": round(aa, 1),
            "winner": home if ha > aa else (away if aa > ha else "Égalité"),
        })

    return {
        "home": home,
        "away": away,
        "home_rating": hr,
        "away_rating": ar,
        "home_win_pct": round(home_win * 100, 1),
        "away_win_pct": round(away_win * 100, 1),
        "draw_pct": round(draw * 100, 1),
        "score_home": max(0, score_home),
        "score_away": max(0, score_away),
        "key_matchups": matchups,
    }


# Canonical name mapping (same as data_loader._TEAM_NAME_MAP)
_LNR_TO_CANONICAL = {
    "Stade Toulousain": "Toulouse",
    "Section Paloise": "Pau",
    "Montpellier Hérault Rugby": "Montpellier",
    "Racing 92": "Racing 92",
    "ASM Clermont": "Clermont",
    "Stade Français Paris": "Paris",
    "Union Bordeaux-Bègles": "Bordeaux",
    "Stade Rochelais": "La Rochelle",
    "RC Toulon": "Toulon",
    "Castres Olympique": "Castres",
    "LOU Rugby": "Lyon",
    "Aviron Bayonnais": "Bayonne",
    "USA Perpignan": "Perpignan",
    "US Montauban": "Montauban",
    "Brive": "Brive",
    "Vannes": "Vannes",
}

_calibration_cache: Optional[dict] = None
_calibration_ts: float = 0
_CALIBRATION_TTL = 3600


def compute_calibration(df: pd.DataFrame, history_path: Path) -> dict:
    """Compute Brier score + accuracy on historical matches."""
    global _calibration_cache, _calibration_ts
    now = time.time()
    if _calibration_cache is not None and now - _calibration_ts < _CALIBRATION_TTL:
        return _calibration_cache

    if not history_path.exists():
        return {"brier_score": None, "accuracy": None, "n_matches": 0}

    with open(history_path, encoding="utf-8") as f:
        matches = json.load(f)

    brier_sum = 0.0
    correct = 0
    n = 0

    for m in matches:
        players = m.get("players", [])
        if not players:
            continue
        home_raw = m.get("home_team", "")
        away_raw = m.get("away_team", "")
        home = _LNR_TO_CANONICAL.get(home_raw, home_raw)
        away = _LNR_TO_CANONICAL.get(away_raw, away_raw)

        home_pts = sum((p.get("points") or 0) for p in players if p.get("side") == "home")
        away_pts = sum((p.get("points") or 0) for p in players if p.get("side") == "away")
        if home_pts == 0 and away_pts == 0:
            continue

        pred = predict_match(df, home, away)
        if "error" in pred:
            continue

        p_home = pred["home_win_pct"] / 100.0
        actual = 1 if home_pts > away_pts else 0
        brier_sum += (p_home - actual) ** 2
        if (p_home >= 0.5) == (actual == 1):
            correct += 1
        n += 1

    if n == 0:
        return {"brier_score": None, "accuracy": None, "n_matches": 0}

    result = {
        "brier_score": round(brier_sum / n, 4),
        "accuracy": round(correct / n * 100, 1),
        "n_matches": n,
    }
    _calibration_cache = result
    _calibration_ts = now
    return result
