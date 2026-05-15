"""Match prediction — reuses the logic from the Streamlit engine."""
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
