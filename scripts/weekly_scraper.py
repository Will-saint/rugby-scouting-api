"""
Weekly LNR scraper — runs in GitHub Actions every Sunday night.
Scrapes Top 14 player stats + match results, updates CSV files in data/seasons/.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SEASON = os.environ.get("SEASON", "2025-2026")
DATA_DIR = Path(__file__).parent.parent / "data"
SEASONS_DIR = DATA_DIR / "seasons" / SEASON
SEASONS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

TEAM_SLUGS = {
    "toulouse": "Toulouse", "racing-92": "Racing 92", "la-rochelle": "La Rochelle",
    "clermont": "Clermont", "bordeaux-begles": "Bordeaux", "montpellier": "Montpellier",
    "paris": "Paris", "toulon": "Toulon", "castres": "Castres", "lyon": "Lyon",
    "bayonne": "Bayonne", "pau": "Pau", "perpignan": "Perpignan", "montauban": "Montauban",
    "brive": "Brive", "vannes": "Vannes",
}


def get_team_stats(slug: str, team_name: str) -> list[dict]:
    """Scrape player stats from LNR team page."""
    url = f"https://top14.lnr.fr/club/{slug}/statistiques/{SEASON}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        comp = soup.find("players-ranking")
        if not comp:
            return []
        ranking = json.loads(comp.get(":ranking", "[]"))
        players = []
        for entry in ranking:
            player_info = entry.get("player", {})
            url_p = player_info.get("url", "")
            # Extract slug and LNR id
            m = re.search(r"/joueur/(\d+)-([^/]+)", url_p)
            if not m:
                continue
            lnr_id, lnr_slug = int(m.group(1)), m.group(2)
            matches = int(entry.get("nbMatchs", 0) or 0)
            minutes_raw = str(entry.get("minutesPlayed", "0") or "0").replace(" ", "").replace("\xa0", "")
            try:
                minutes = float(minutes_raw)
            except ValueError:
                minutes = 0.0
            players.append({
                "lnr_id": lnr_id,
                "lnr_slug": lnr_slug,
                "name": player_info.get("name", "").title(),
                "team": team_name,
                "season": SEASON,
                "photo_url": (player_info.get("image") or {}).get("original"),
                "matches_played": matches,
                "minutes_total": minutes,
                "points_scored_total": float(entry.get("nbPoints") or 0),
                "tries_total": float(entry.get("nbEssais") or 0),
                "offloads_total": float(entry.get("offload") or 0),
                "line_breaks_total": float(entry.get("lineBreak") or 0),
                "turnovers_won_total": float(entry.get("breakdownSteals") or 0),
                "tackles_success_total": float(entry.get("totalSuccessfulTackles") or 0),
                "yellow_cards": int(entry.get("nbCartonsJaunes") or 0),
                "orange_cards": int(entry.get("nbCartonsOranges") or 0),
                "red_cards": int(entry.get("nbCartonsRouges") or 0),
            })
        print(f"  {team_name}: {len(players)} players")
        return players
    except Exception as e:
        print(f"  ERROR {team_name}: {e}")
        return []


def compute_per80(p: dict) -> dict:
    mins = p.get("minutes_total") or 0
    if mins <= 0:
        return p
    def per80(field):
        v = p.get(field)
        return round(v / mins * 80, 2) if v is not None else None
    p["tackles_per80"] = per80("tackles_success_total")
    p["offloads_per80"] = per80("offloads_total")
    p["line_breaks_per80"] = per80("line_breaks_total")
    p["turnovers_won_per80"] = per80("turnovers_won_total")
    p["tries_per80"] = per80("tries_total")
    p["points_scored_per80"] = per80("points_scored_total")
    return p


def update_match_history():
    """Scrape recent match results and update lnr_match_history.json."""
    hist_path = DATA_DIR / "lnr_match_history.json"
    existing = []
    if hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing_ids = {m.get("fixture_id") for m in existing}

    # Get current round matches from classement page
    url = f"https://top14.lnr.fr/classement/{SEASON}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    slider = soup.find("score-slider")
    if not slider:
        print("  No score-slider found")
        return

    matches_raw = json.loads(slider.get(":matches", "[]"))
    new_matches = 0
    for m in matches_raw:
        if m.get("status") not in ("played", "finished") or m.get("id") in existing_ids:
            continue
        score = m.get("score", [0, 0])
        home_club = m.get("hosting_club", {}).get("name", "")
        away_club = m.get("visiting_club", {}).get("name", "")
        existing.append({
            "fixture_id": m.get("id"),
            "round": m.get("week", ""),
            "season": SEASON,
            "date": m.get("date", ""),
            "home_team": home_club,
            "away_team": away_club,
            "home_score": score[0] if len(score) > 0 else 0,
            "away_score": score[1] if len(score) > 1 else 0,
            "players": [],  # Player-level stats not available from score-slider
        })
        new_matches += 1

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)
    print(f"  Match history: {len(existing)} total, {new_matches} new")


def main():
    print(f"=== Weekly LNR Scraper — {SEASON} ===")

    all_players = []
    for slug, team_name in TEAM_SLUGS.items():
        players = get_team_stats(slug, team_name)
        all_players.extend(players)
        time.sleep(0.5)

    print(f"\nTotal: {len(all_players)} players scraped")

    # Compute per80 stats
    all_players = [compute_per80(p) for p in all_players]

    # Save as simple CSV (will be enriched by ratings pipeline when run locally)
    import pandas as pd
    df = pd.DataFrame(all_players)
    raw_path = SEASONS_DIR / "players_raw_lnr.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw: {raw_path}")

    # Update match history
    print("\nUpdating match history...")
    update_match_history()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
