"""
AI commentary + scout recommendations via Claude Haiku.
Requires ANTHROPIC_API_KEY env var — graceful no-op if absent.
"""
import os
import time
from typing import Optional

_client = None
_commentary_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 3600  # 1h


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            try:
                import anthropic
                _client = anthropic.Anthropic(api_key=key)
            except ImportError:
                pass
    return _client


def _call(prompt: str, max_tokens: int = 220) -> Optional[str]:
    client = _get_client()
    if not client:
        return None
    try:
        import anthropic
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[AI] {e}")
        return None


def generate_commentary(player: dict, season: str) -> Optional[str]:
    """3-sentence expert analysis of a player's current season stats."""
    cache_key = f"cmt_{player.get('lnr_slug')}_{season}"
    now = time.time()
    if cache_key in _commentary_cache:
        text, ts = _commentary_cache[cache_key]
        if now - ts < _CACHE_TTL:
            return text

    trend = player.get("form_trend", "→")
    form_label = (
        "en nette progression" if "↗" in str(trend)
        else "en perte de vitesse" if "↘" in str(trend)
        else "stable"
    )
    form_score = player.get("form_score")
    intl_line = ""
    if player.get("rating_intl"):
        intl_line = f"- International : {player.get('matches_intl')} sélections pour {player.get('team_intl')}"

    prompt = f"""Tu es un analyste rugby de Top 14. Rédige exactement 3 phrases percutantes en français sur ce joueur.

Joueur : {player.get('name')} — {player.get('position_label') or player.get('position_group')} — {player.get('team')}
Note : {player.get('rating')}/99 (tier {player.get('tier')}) | Forme : {form_label}{f' ({round(form_score)}/100)' if form_score else ''}
Stats par 80 min : plaquages={player.get('tackles_per80') or '—'}, offloads={player.get('offloads_per80') or '—'}, franchissements={player.get('line_breaks_per80') or '—'}, turnovers récupérés={player.get('turnovers_won_per80') or '—'}, essais={player.get('tries_per80') or '—'}
{intl_line}

Règles strictes : exactement 3 phrases. Phrase 1 = état de forme actuel. Phrase 2 = point fort distinctif. Phrase 3 = point d'attention ou perspective recrutement. Ton analyste expert, pas de superlatifs vides. Reformule les stats en qualitatif, ne donne pas les chiffres bruts."""

    text = _call(prompt, max_tokens=220)
    if text:
        _commentary_cache[cache_key] = (text, now)
    return text


def generate_scout_summary(players: list[dict], criteria: dict) -> Optional[str]:
    """2-3 sentence summary of why these players match the scouting criteria."""
    pos_label = criteria.get("position_label", criteria.get("position", "tous postes"))
    min_r = criteria.get("min_rating", "—")
    max_r = criteria.get("max_rating", "—")

    profiles = "\n".join(
        f"- {p['name']} ({p['team']}, {p.get('position_label') or p.get('position_group')}) : {p['rating']}/99"
        for p in players[:5]
    )

    prompt = f"""Tu es recruteur rugby professionnel. En 2 phrases maximum, justifie pourquoi ces joueurs correspondent au profil recherché.

Profil recherché : {pos_label}, note cible {min_r}–{max_r}/99
Profils identifiés :
{profiles}

Règles : 2 phrases max, ton de recruteur expert et direct. Cite les 1-2 noms les plus remarquables. Mentionne ce qui les distingue pour le recrutement."""

    return _call(prompt, max_tokens=150)
