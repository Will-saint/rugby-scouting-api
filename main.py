"""FastAPI — Rugby Analytics Dashboard API."""
import io
import os
from typing import Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from data_loader import (
    get_df, get_seasons, row_to_summary, row_to_detail,
    rating_to_tier, safe_float, safe_int, safe_str, DEFAULT_SEASON, AVAILABLE_SEASONS,
    get_player_badges,
)
from predictor import predict_match
from ai_service import generate_commentary, generate_scout_summary

limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])
app = FastAPI(title="Rugby Analytics API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",
]
if os.environ.get("ALLOWED_ORIGIN"):
    _ORIGINS.append(os.environ["ALLOWED_ORIGIN"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1"


# ── Seasons & Meta ────────────────────────────────────────────────────────────

@app.get(f"{PREFIX}/seasons")
def seasons():
    return get_seasons()


@app.get(f"{PREFIX}/meta")
def meta(season: str = DEFAULT_SEASON):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")
    top = df.nlargest(1, "rating").iloc[0]
    return {
        "season": season,
        "n_players": len(df),
        "n_teams": int(df["team"].nunique()),
        "avg_rating": round(float(df["rating"].mean()), 1),
        "top_player": {"name": top.get("name"), "rating": safe_float(top.get("rating"))},
        "last_updated": None,
    }


# ── Players ───────────────────────────────────────────────────────────────────

@app.get(f"{PREFIX}/players")
def players(
    season: str = DEFAULT_SEASON,
    position: Optional[str] = None,
    team: Optional[str] = None,
    min_rating: Optional[float] = None,
    limit: int = Query(50, le=544),
    offset: int = 0,
):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    if position:
        df = df[df["position_group"] == position]
    if team:
        df = df[df["team"] == team]
    if min_rating is not None:
        df = df[df["rating"] >= min_rating]

    df = df.sort_values("rating", ascending=False)
    total = len(df)
    chunk = df.iloc[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "players": [row_to_summary(r) for _, r in chunk.iterrows()]}


@app.get(f"{PREFIX}/players/search")
def search_players(q: str, season: str = DEFAULT_SEASON, limit: int = 20):
    df = get_df(season)
    if df.empty:
        return []
    mask = df["name"].str.contains(q, case=False, na=False)
    results = df[mask].sort_values("rating", ascending=False).head(limit)
    return [row_to_summary(r) for _, r in results.iterrows()]


@app.get(f"{PREFIX}/players/{{lnr_slug}}/history")
def player_history(lnr_slug: str):
    history = []
    for s in AVAILABLE_SEASONS:
        df = get_df(s)
        if df.empty:
            continue
        match = df[df["lnr_slug"] == lnr_slug]
        if match.empty:
            continue
        row = match.iloc[0]
        history.append({
            "season": s,
            "rating": safe_float(row.get("rating")),
            "age": safe_int(row.get("age")),
            "minutes_played": safe_int(row.get("minutes_played")),
        })
    return history


@app.get(f"{PREFIX}/players/{{lnr_slug}}")
def player_detail(lnr_slug: str, season: str = DEFAULT_SEASON):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")
    match = df[df["lnr_slug"] == lnr_slug]
    if match.empty:
        raise HTTPException(404, f"Player '{lnr_slug}' not found in {season}")
    row = match.iloc[0]
    history = player_history(lnr_slug)
    return row_to_detail(row, history)


@app.get(f"{PREFIX}/players/{{lnr_slug}}/commentary")
@limiter.limit("10/minute")
def player_commentary(request: Request, lnr_slug: str, season: str = DEFAULT_SEASON):
    """Generate a 3-sentence AI analysis of a player's current season."""
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")
    match = df[df["lnr_slug"] == lnr_slug]
    if match.empty:
        raise HTTPException(404, f"Player '{lnr_slug}' not found")
    row = match.iloc[0]
    player_data = row_to_detail(row, [])
    text = generate_commentary(player_data, season)
    if text is None:
        return {"commentary": None, "available": False}
    return {"commentary": text, "available": True}


# ── Scout IA ──────────────────────────────────────────────────────────────────

POSITION_LABELS_FR = {
    "FRONT_ROW": "Première ligne", "LOCK": "Deuxième ligne",
    "BACK_ROW": "Troisième ligne", "SCRUM_HALF": "Demi de mêlée",
    "FLY_HALF": "Demi d'ouverture", "WINGER": "Ailier",
    "CENTRE": "Centre", "FULLBACK": "Arrière",
}

@app.get(f"{PREFIX}/scout")
@limiter.limit("20/minute")
def scout(
    request: Request,
    season: str = DEFAULT_SEASON,
    position: Optional[str] = None,
    min_rating: float = 60.0,
    max_rating: float = 99.0,
    exclude_team: Optional[str] = None,
    limit: int = Query(10, le=30),
    with_ai: bool = True,
):
    """Find best players matching scouting criteria + optional AI summary."""
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    mask = (df["rating"] >= min_rating) & (df["rating"] <= max_rating)
    if position and position != "ALL":
        mask &= df["position_group"] == position
    if exclude_team:
        mask &= df["team"] != exclude_team

    results = df[mask].sort_values("rating", ascending=False).head(limit)
    players = [row_to_summary(r) for _, r in results.iterrows()]

    ai_summary = None
    if with_ai and players:
        criteria = {
            "position": position or "tous postes",
            "position_label": POSITION_LABELS_FR.get(position or "", position or "tous postes"),
            "min_rating": round(min_rating, 1),
            "max_rating": round(max_rating, 1),
        }
        ai_summary = generate_scout_summary(players, criteria)

    return {
        "players": players,
        "total": len(players),
        "criteria": {
            "position": position, "min_rating": min_rating,
            "max_rating": max_rating, "exclude_team": exclude_team, "season": season,
        },
        "ai_summary": ai_summary,
        "ai_available": ai_summary is not None,
    }


# ── Teams ─────────────────────────────────────────────────────────────────────

@app.get(f"{PREFIX}/teams")
def teams(season: str = DEFAULT_SEASON):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    result = []
    for team, group in df.groupby("team"):
        avg = float(group["rating"].mean())
        result.append({
            "team": team,
            "avg_rating": round(avg, 1),
            "tier": rating_to_tier(avg),
            "n_players": len(group),
            "top_player": group.nlargest(1, "rating").iloc[0].get("name"),
        })
    result.sort(key=lambda x: -x["avg_rating"])
    for i, t in enumerate(result):
        t["rank"] = i + 1
    return result


@app.get(f"{PREFIX}/teams/{{team_name}}")
def team_detail(team_name: str, season: str = DEFAULT_SEASON):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    t = df[df["team"] == team_name]
    if t.empty:
        raise HTTPException(404, f"Team '{team_name}' not found")

    avg = float(t["rating"].mean())
    axis_cols = ["axis_att", "axis_def", "axis_ctrl", "axis_kick", "axis_pow", "axis_gabarit"]
    axes = {col: round(float(t[col].mean()), 1) for col in axis_cols if col in t.columns}

    tier_dist = {}
    for _, row in t.iterrows():
        tier = rating_to_tier(safe_float(row.get("rating")))
        tier_dist[tier] = tier_dist.get(tier, 0) + 1

    roster_by_pos = {}
    for pos, group in t.groupby("position_group"):
        roster_by_pos[pos] = [row_to_summary(r) for _, r in group.sort_values("rating", ascending=False).iterrows()]

    top5 = [row_to_summary(r) for _, r in t.nlargest(5, "rating").iterrows()]

    # Team history across seasons
    history = []
    for s in AVAILABLE_SEASONS:
        sdf = get_df(s)
        if sdf.empty:
            continue
        st = sdf[sdf["team"] == team_name]
        if st.empty:
            continue
        history.append({"season": s, "avg_rating": round(float(st["rating"].mean()), 1), "n_players": len(st)})

    return {
        "team": team_name,
        "season": season,
        "avg_rating": round(avg, 1),
        "tier": rating_to_tier(avg),
        "n_players": len(t),
        "axes": axes,
        "tier_distribution": tier_dist,
        "roster_by_position": roster_by_pos,
        "top5": top5,
        "history": history,
    }


# ── Leaderboard ───────────────────────────────────────────────────────────────

@app.get(f"{PREFIX}/leaderboard")
def leaderboard(
    season: str = DEFAULT_SEASON,
    position: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = Query(100, le=544),
):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    if position and position != "ALL":
        df = df[df["position_group"] == position]
    if team and team != "ALL":
        df = df[df["team"] == team]

    df = df.sort_values("rating", ascending=False).head(limit).reset_index(drop=True)
    result = []
    for i, (_, row) in enumerate(df.iterrows()):
        rating = safe_float(row.get("rating")) or 40.0
        lnr_slug = safe_str(row.get("lnr_slug")) or ""
        team = safe_str(row.get("team")) or ""
        season_val = safe_str(row.get("season")) or season
        result.append({
            "rank":             i + 1,
            "lnr_slug":         lnr_slug,
            "name":             safe_str(row.get("name")),
            "team":             team,
            "position_group":   safe_str(row.get("position_group")),
            "rating":           round(rating, 1),
            "tier":             rating_to_tier(rating),
            "age":              safe_int(row.get("age")),
            "nationality":      safe_str(row.get("nationality")),
            "form_trend":       safe_str(row.get("form_trend")) or "→",
            "confidence_badge": safe_str(row.get("confidence_badge")) or "Basse",
            "badges":           get_player_badges(lnr_slug, team, season_val),
        })
    return result


@app.get(f"{PREFIX}/leaderboard/export")
@limiter.limit("30/hour")
def leaderboard_export(
    request: Request,
    season: str = DEFAULT_SEASON,
    position: Optional[str] = None,
    team: Optional[str] = None,
):
    """Export leaderboard as CSV file."""
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    if position and position != "ALL":
        df = df[df["position_group"] == position]
    if team and team != "ALL":
        df = df[df["team"] == team]

    df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    export_cols = [
        "rank", "name", "team", "position_group", "position_label",
        "rating", "tier", "age", "nationality",
        "tackles_per80", "offloads_per80", "line_breaks_per80",
        "turnovers_won_per80", "tries_per80", "kick_points_per80",
        "rating_intl", "team_intl", "matches_intl",
        "minutes_total", "matches_played", "season",
    ]
    cols = [c for c in export_cols if c in df.columns]
    csv_buf = io.StringIO()
    df[cols].to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    filename = f"top14_{season}_ratings"
    if position and position != "ALL":
        filename += f"_{position.lower()}"
    if team and team != "ALL":
        filename += f"_{team.lower().replace(' ', '_')}"
    filename += ".csv"

    return StreamingResponse(
        iter([csv_buf.read()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Composition XV ───────────────────────────────────────────────────────────

POS_WEIGHT = {
    "FRONT_ROW": 1.0, "LOCK": 1.0, "BACK_ROW": 1.1,
    "SCRUM_HALF": 1.2, "FLY_HALF": 1.3,
    "WINGER": 1.0, "CENTRE": 1.1, "FULLBACK": 1.2,
}

class CompositionBody(BaseModel):
    slugs: list[str]   # liste de lnr_slug (15 max)
    season: str = DEFAULT_SEASON


@app.post(f"{PREFIX}/composition/score")
def composition_score(body: CompositionBody):
    df = get_df(body.season)
    if df.empty:
        raise HTTPException(404, "Season not found")

    players_out = []
    total_w = total_wr = 0.0
    axis_sums: dict[str, float] = {}
    axis_counts: dict[str, int] = {}
    axis_cols = ["axis_att", "axis_def", "axis_ctrl", "axis_kick", "axis_pow", "axis_gabarit", "axis_disc"]

    for slug in body.slugs[:15]:
        match = df[df["lnr_slug"] == slug]
        if match.empty:
            continue
        row = match.iloc[0]
        pg = row.get("position_group", "FRONT_ROW")
        w = POS_WEIGHT.get(pg, 1.0)
        rating = safe_float(row.get("rating")) or 40.0
        total_w  += w
        total_wr += w * rating
        for ax in axis_cols:
            v = safe_float(row.get(ax))
            if v is not None:
                axis_sums[ax]   = axis_sums.get(ax, 0.0) + v
                axis_counts[ax] = axis_counts.get(ax, 0) + 1
        players_out.append(row_to_summary(row))

    collective = round(total_wr / total_w, 1) if total_w > 0 else 0.0
    axes_avg = {ax: round(axis_sums[ax] / axis_counts[ax], 1) for ax in axis_sums}

    return {
        "collective_score": collective,
        "tier": rating_to_tier(collective),
        "n_players": len(players_out),
        "axes": axes_avg,
        "players": players_out,
    }


# ── International ────────────────────────────────────────────────────────────

@app.get(f"{PREFIX}/international")
def international(season: str = DEFAULT_SEASON):
    df = get_df(season)
    if df.empty:
        raise HTTPException(404, "Season not found")
    if "rating_intl" not in df.columns:
        return []
    intl = df[df["rating_intl"].notna()].copy()
    intl = intl.sort_values("rating_intl", ascending=False)
    result = []
    for _, row in intl.iterrows():
        rating = safe_float(row.get("rating")) or 40.0
        rating_intl = safe_float(row.get("rating_intl"))
        result.append({
            "lnr_slug":       safe_str(row.get("lnr_slug")),
            "name":           safe_str(row.get("name")),
            "team":           safe_str(row.get("team")),
            "position_group": safe_str(row.get("position_group")),
            "rating":         round(rating, 1),
            "tier":           rating_to_tier(rating),
            "rating_intl":    round(rating_intl, 1) if rating_intl else None,
            "tier_intl":      rating_to_tier(rating_intl),
            "team_intl":      safe_str(row.get("team_intl")),
            "matches_intl":   safe_int(row.get("matches_intl")),
            "nationality":    safe_str(row.get("nationality")),
        })
    return result


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictBody(BaseModel):
    home: str
    away: str
    season: str = DEFAULT_SEASON


@app.post(f"{PREFIX}/predict")
def predict(body: PredictBody):
    df = get_df(body.season)
    if df.empty:
        raise HTTPException(404, "Season not found")
    result = predict_match(df, body.home, body.away)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result
