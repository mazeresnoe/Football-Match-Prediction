import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

# Ligues à scraper — clé API : nom lisible
LEAGUES = {
    "soccer_france_ligue_one":       "Ligue 1",
    "soccer_france_ligue_two":       "Ligue 2",
    "soccer_spain_la_liga":          "La Liga",
    "soccer_spain_segunda_division": "La Liga 2",
    "soccer_italy_serie_a":          "Serie A",
    "soccer_italy_serie_b":          "Serie B",
    "soccer_germany_bundesliga":     "Bundesliga",
    "soccer_germany_bundesliga2":    "Bundesliga 2",
    "soccer_epl":                    "Premier League",
    "soccer_efl_champ":              "Championship",
    "soccer_uefa_champs_league":     "Champions League",
    "soccer_uefa_europa_league":     "Europa League",
    "soccer_uefa_europa_conference_league": "Conference League",
    # FIFA Club World Cup pas dispo sur The Odds API pour l'instant
}

def get_season(commence_time: str) -> str:
    """Détermine la saison à partir de la date du match."""
    dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    year = dt.year
    month = dt.month
    if month >= 7:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"

# Bookmakers cibles — clés telles que retournées par The Odds API
TARGET_BOOKMAKERS = {
    "unibet_fr":  "Unibet",
    "betclic_fr": "Betclic",
    "winamax_fr": "Winamax",
}

def get_target_odds(bookmakers: list, home_team: str, away_team: str) -> dict:
    """
    Pour chaque bookmaker cible, extrait les cotes 1/N/2.
    Retourne un dict:
    {
        "Unibet":  {"home": 1.85, "draw": 3.40, "away": 4.20},
        "Betclic": {"home": 1.90, "draw": 3.50, "away": 4.00},
        "Winamax": {"home": 1.88, "draw": 3.45, "away": 4.10},
    }
    Valeur None si bookmaker absent ou cote manquante.
    """
    result = {name: {"home": None, "draw": None, "away": None}
              for name in TARGET_BOOKMAKERS.values()}

    for bookie in bookmakers:
        bookie_key = bookie.get("key", "")
        if bookie_key not in TARGET_BOOKMAKERS:
            continue

        bookie_label = TARGET_BOOKMAKERS[bookie_key]

        for market in bookie.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market["outcomes"]:
                name = outcome["name"]
                price = outcome["price"]
                if name == "Draw":
                    result[bookie_label]["draw"] = price
                elif name == home_team:
                    result[bookie_label]["home"] = price
                elif name == away_team:
                    result[bookie_label]["away"] = price

    return result

def fetch_odds_for_league(sport_key: str, league_name: str) -> list:
    """Récupère les cotes pour une ligue et retourne une liste de rows CSV."""
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "bookmakers": ",".join(TARGET_BOOKMAKERS.keys()),
    }

    response = requests.get(url, params=params)

    remaining = response.headers.get("x-requests-remaining", "?")
    used = response.headers.get("x-requests-used", "?")
    print(f"  [{league_name}] Requêtes utilisées: {used} | Restantes: {remaining}")

    if response.status_code != 200:
        print(f"  ERREUR {response.status_code} pour {league_name}: {response.text}")
        return []

    matches = response.json()
    rows = []

    for match in matches:
        commence_time = match.get("commence_time", "")
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        bookmakers = match.get("bookmakers", [])

        odds = get_target_odds(bookmakers, home_team, away_team)
        season = get_season(commence_time)
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")

        rows.append({
            "date":                  date_str,
            "home_team":             home_team,
            "away_team":             away_team,
            "unibet_home":           odds["Unibet"]["home"],
            "unibet_draw":           odds["Unibet"]["draw"],
            "unibet_away":           odds["Unibet"]["away"],
            "betclic_home":          odds["Betclic"]["home"],
            "betclic_draw":          odds["Betclic"]["draw"],
            "betclic_away":          odds["Betclic"]["away"],
            "winamax_home":          odds["Winamax"]["home"],
            "winamax_draw":          odds["Winamax"]["draw"],
            "winamax_away":          odds["Winamax"]["away"],
            "league_code":           sport_key,
            "season":                season,
        })

    return rows

def main():
    if not API_KEY:
        print("ERREUR: ODDS_API_KEY non trouvée dans .env")
        return

    all_rows = []
    print("Début du scraping...\n")

    for sport_key, league_name in LEAGUES.items():
        print(f"Scraping: {league_name}...")
        rows = fetch_odds_for_league(sport_key, league_name)
        print(f"  {len(rows)} matchs récupérés\n")
        all_rows.extend(rows)

    if not all_rows:
        print("Aucune donnée récupérée. Vérifie ta clé API et ta connexion.")
        return

    # Sauvegarde CSV
    output_file = "data/futur_odds/upcoming_odds.csv"
    os.makedirs("data/futur_odds", exist_ok=True)

    fieldnames = [
        "date", "home_team", "away_team",
        "unibet_home", "unibet_draw", "unibet_away",
        "betclic_home", "betclic_draw", "betclic_away",
        "winamax_home", "winamax_draw", "winamax_away",
        "league_code", "season",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✓ {len(all_rows)} matchs sauvegardés dans {output_file}")

if __name__ == "__main__":
    main()