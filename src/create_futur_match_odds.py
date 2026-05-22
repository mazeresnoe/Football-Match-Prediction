"""
Script pour créer le dataset futur_match_odds.
Prend tous les matchs "Not started" ET dans le futur,
les merge avec les cotes futures normalisées.
Garde uniquement les matchs qui ont des cotes disponibles.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

EVENT_IDS_DIR = Path("data/raw/event_ids")
ODDS_PATH     = Path("data/futur_odds/upcoming_odds_normalized.csv")
OUTPUT_PATH   = Path("data/futur_match_odds/futur_match_odds.csv")

LEAGUE_CODE_MAP = {
    "premier_league":         "soccer_epl",
    "championship":           "soccer_efl_champ",
    "ligue_1":                "soccer_france_ligue_one",
    "ligue_2":                "soccer_france_ligue_two",
    "la_liga":                "soccer_spain_la_liga",
    "la_liga_2":              "soccer_spain_segunda_division",
    "serie_a":                "soccer_italy_serie_a",
    "serie_b":                "soccer_italy_serie_b",
    "bundesliga":             "soccer_germany_bundesliga",
    "2_bundesliga":           "soccer_germany_bundesliga2",
    "UEFA_champions_league":  "soccer_uefa_champs_league",
    "UEFA_europa_league":     "soccer_uefa_europa_league",
    "UEFA_conference_league": "soccer_uefa_europa_conference_league",
    "Fifa_world_cup_club":    None,
}


# ============================================================
# ÉTAPE 1 — Charger les matchs futurs "Not started"
# ============================================================

def load_upcoming_events() -> pd.DataFrame:
    now = datetime.now()
    rows = []

    for league_folder in sorted(EVENT_IDS_DIR.iterdir()):
        if not league_folder.is_dir():
            continue
        league_name = league_folder.name
        league_code = LEAGUE_CODE_MAP.get(league_name)

        for json_file in sorted(league_folder.glob("*.json")):
            season = json_file.stem
            try:
                data = json.load(open(json_file, encoding="utf-8"))
            except Exception as e:
                print(f"  ⚠ Erreur lecture {json_file}: {e}")
                continue

            for match in data:
                # Filtre 1 : statut "Not started" uniquement
                if match.get("description") != "Not started":
                    continue

                # Filtre 2 : date strictement dans le futur
                raw_date = match.get("date")
                if not raw_date:
                    continue
                try:
                    match_date = datetime.fromisoformat(raw_date)
                    if match_date <= now:
                        continue
                except Exception:
                    continue

                rows.append({
                    "event_id":      match.get("event_id"),
                    "date":          raw_date,
                    "home_team":     match.get("home_team", "").lower().strip(),
                    "away_team":     match.get("away_team", "").lower().strip(),
                    "round":         match.get("round"),
                    "tournament_id": match.get("tournament_id"),
                    "league":        league_name,
                    "league_code":   league_code,
                    "season":        season,
                })

    df = pd.DataFrame(rows)
    print(f"✓ {len(df)} matchs futurs 'Not started' chargés")
    return df


# ============================================================
# ÉTAPE 2 — Charger les cotes futures normalisées
# ============================================================

def load_odds() -> pd.DataFrame:
    df = pd.read_csv(ODDS_PATH)
    df["home_team"] = df["home_team"].str.lower().str.strip()
    df["away_team"] = df["away_team"].str.lower().str.strip()
    df["date_merge"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    print(f"✓ {len(df)} matchs chargés depuis upcoming_odds_normalized.csv")
    return df


# ============================================================
# ÉTAPE 3 — Merge + filtre sur cotes disponibles
# ============================================================

def merge(events: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    events["date_merge"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")

    odds_cols = [
        "home_team", "away_team", "date_merge", "league_code",
        "unibet_home", "unibet_draw", "unibet_away",
        "betclic_home", "betclic_draw", "betclic_away",
        "winamax_home", "winamax_draw", "winamax_away",
    ]

    merged = events.merge(
        odds[odds_cols],
        on=["home_team", "away_team", "date_merge", "league_code"],
        how="left",
    )

    total = len(merged)
    with_odds = merged["unibet_home"].notna().sum()
    without   = merged["unibet_home"].isna().sum()

    print(f"\n📊 Avant filtre :")
    print(f"   • Avec cotes    : {with_odds} / {total}")
    print(f"   • Sans cotes    : {without} (trop loin ou hors couverture API)")

    # Garde uniquement les matchs avec au moins une cote disponible
    merged = merged.dropna(subset=["unibet_home", "betclic_home", "winamax_home"], how="all")
    merged = merged.drop(columns=["date_merge"]).reset_index(drop=True)

    print(f"   • Après filtre  : {len(merged)} matchs avec cotes")
    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CRÉATION DU DATASET FUTUR_MATCH_ODDS")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    events = load_upcoming_events()
    if events.empty:
        print("❌ Aucun match futur 'Not started' trouvé.")
        return

    odds = load_odds()
    merged = merge(events, odds)

    if merged.empty:
        print("❌ Aucun match avec cotes disponibles.")
        return

    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Dataset sauvegardé : {OUTPUT_PATH}")
    print(f"   {len(merged)} matchs avec cotes\n")
    print(merged[["date", "league", "home_team", "away_team",
                  "unibet_home", "unibet_draw", "unibet_away"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()