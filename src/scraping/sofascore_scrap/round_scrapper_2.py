
"""
Version modifiée du EventIdScraper qui scrappe UNIQUEMENT la saison en cours.
Évite de re-scraper les anciennes saisons avec des matchs non joués.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from playwright.sync_api import sync_playwright
import sys
import random

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
from src.utils.mapping.map import ALL_LEAGUES, SEASONS_BY_LEAGUE, MAX_ROUNDS_BY_LEAGUE_AND_SEASON

# ============================================================
# SAISONS EN COURS PAR CATÉGORIE
# Modifie ici quand une nouvelle saison commence
# ============================================================
CURRENT_SEASONS = {
    'TOP_5_AND_D2':            '2025/2026',
    'EUROPEAN_COMPETITIONS':   '2025/2026',
    'INTERNATIONAL_COMPETITIONS': '2024/2025',  # ajuste si besoin
}

# Exception pour Fifa_world_cup_club qui a un format différent
CURRENT_SEASON_OVERRIDES = {
    'Fifa_world_cup_club': '2025',
}


class EventIdScraperCurrentSeason:
    def __init__(self):
        self.base_path = Path("data/raw/event_ids")

    def fetch_rounds_info(self, tournament_id, season_id):
        rounds_info = []
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            def handle_response(response):
                if url in response.url and response.status == 200:
                    try:
                        data = response.json()
                        for rnd in data.get("rounds", []):
                            round_number = rnd.get("round")
                            if round_number is not None:
                                rounds_info.append({
                                    "round_number": round_number,
                                    "name": rnd.get("name"),
                                    "slug": rnd.get("slug"),
                                    "prefix": rnd.get("prefix")
                                })
                    except Exception as e:
                        print(f"[WARNING] Error parsing rounds response: {e}")

            page.on("response", handle_response)
            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(1500)
            except Exception as e:
                print(f"[ERROR] Could not fetch rounds info: {e}")
            finally:
                browser.close()

        valid_rounds = [r for r in rounds_info if r['round_number'] is not None]
        return sorted(valid_rounds, key=lambda x: x['round_number']) if valid_rounds else []

    def get_rounds_with_fallback(self, tournament_id, season_id, league_name, saison):
        print(f"🔍 Fetching rounds info from API...")
        api_rounds = self.fetch_rounds_info(tournament_id, season_id)

        if api_rounds:
            print(f"✅ Found {len(api_rounds)} rounds from API")
            return api_rounds

        print(f"⚠️  API failed, using fallback mapping...")
        max_round = MAX_ROUNDS_BY_LEAGUE_AND_SEASON.get(league_name, {}).get(saison)
        if max_round:
            fallback_rounds = [
                {"round_number": i, "slug": None, "prefix": None}
                for i in range(1, max_round + 1)
            ]
            print(f"📋 Using fallback: {len(fallback_rounds)} rounds")
            return fallback_rounds

        print(f"🚨 No mapping found, using default 38 rounds")
        return [{"round_number": i, "slug": None, "prefix": None} for i in range(1, 39)]

    def scrape_events_for_round(self, tournament_id, season_id, round_info):
        round_number = round_info['round_number']
        base_url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_number}"

        if round_info.get("slug"):
            if round_info.get("prefix"):
                url = f"{base_url}/slug/{round_info['prefix']}/{round_info['slug']}"
            else:
                url = f"{base_url}/slug/{round_info['slug']}"
        else:
            url = base_url

        events_info = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            def handle_response(response):
                if response.status == 200 and f"/events/round/{round_number}" in response.url:
                    try:
                        json_data = response.json()
                        events = json_data.get("events", [])
                        for match in events:
                            round_info_data = match.get("roundInfo", {})
                            events_info.append({
                                "event_id": match.get("id"),
                                "home_team": match.get("homeTeam", {}).get("name"),
                                "away_team": match.get("awayTeam", {}).get("name"),
                                "date": datetime.fromtimestamp(match.get("startTimestamp")).isoformat() if match.get("startTimestamp") else None,
                                "round": round_info_data.get("round", round_number),
                                "round_name": round_info_data.get("name"),
                                "round_slug": round_info_data.get("slug"),
                                "tournament_id": tournament_id,
                                "season_id": season_id,
                                "description": match.get("status", {}).get("description"),
                                "api_slug": round_info.get("slug"),
                                "api_prefix": round_info.get("prefix"),
                                "homeScore": (match.get("homeScore", {}) or {}).get("current"),
                                "awayScore": (match.get("awayScore", {}) or {}).get("current")
                            })
                    except Exception as e:
                        print(f"[WARNING] Error parsing events response: {e}")

            page.on("response", handle_response)
            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"[ERROR] Round {round_number} failed: {e}")
            finally:
                browser.close()

        return events_info

    def merge_events(self, existing_events, new_events):
        if not existing_events:
            return new_events
        if not new_events:
            return existing_events

        existing_by_id = {event['event_id']: event for event in existing_events}
        merged_events = []
        processed_ids = set()

        for new_event in new_events:
            event_id = new_event['event_id']
            processed_ids.add(event_id)

            if event_id in existing_by_id:
                old_event = existing_by_id[event_id]
                old_status = old_event.get('description')
                new_status = new_event.get('description')

                if old_status == 'Ended' and new_status != 'Ended':
                    merged_events.append(old_event)
                else:
                    merged_events.append(new_event)
            else:
                merged_events.append(new_event)

        for existing_event in existing_events:
            if existing_event['event_id'] not in processed_ids:
                merged_events.append(existing_event)

        return merged_events

    def analyze_existing_events(self, existing_events):
        rounds_analysis = {}
        if not existing_events:
            return rounds_analysis

        round_match_status = defaultdict(list)
        for event in existing_events:
            round_match_status[event.get("round")].append(event.get("description"))

        FINAL_STATUSES = {"Ended", "Canceled"}

        for rnd, descriptions in round_match_status.items():
            if descriptions:
                final_count = sum(1 for desc in descriptions if desc in FINAL_STATUSES)
                total_count = len(descriptions)
                rounds_analysis[rnd] = {
                    "total": total_count,
                    "ended": final_count,
                    "all_ended": final_count == total_count,
                    "has_non_ended": final_count < total_count
                }

        return rounds_analysis

    def save_events_to_json(self, league, saison, events):
        safe_saison = saison.replace("/", "-")
        dir_path = self.base_path / league
        file_path = dir_path / f"{safe_saison}.json"

        os.makedirs(dir_path, exist_ok=True)
        sorted_events = sorted(events, key=lambda e: (e['round'], e['date'] or ''))

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sorted_events, f, indent=2, ensure_ascii=False)

    def get_current_season(self, category: str, league_name: str) -> str | None:
        """Retourne la saison en cours pour une ligue donnée."""
        # Override spécifique par ligue
        if league_name in CURRENT_SEASON_OVERRIDES:
            return CURRENT_SEASON_OVERRIDES[league_name]
        # Saison par défaut pour la catégorie
        return CURRENT_SEASONS.get(category)

    def scrape_current_season_only(self):
        """
        Scrappe UNIQUEMENT la saison en cours pour chaque ligue.
        Ignore toutes les saisons passées.
        """
        for league_category, leagues in ALL_LEAGUES.items():
            print(f"\n🏆 Category: {league_category}")

            # Ignore les compétitions internationales (pas dans SEASONS_BY_LEAGUE)
            if league_category not in SEASONS_BY_LEAGUE:
                print(f"  ⏭ Skipping {league_category} (not in SEASONS_BY_LEAGUE)")
                continue

            for league_name, league_data in leagues.items():
                tournament_id = league_data['tournament_id']
                seasons_dict = SEASONS_BY_LEAGUE.get(league_category, {}).get(league_name, {})

                if not seasons_dict:
                    print(f"  ⏭ {league_name}: no seasons found, skipping")
                    continue

                # ← FILTRE CLÉ : on prend uniquement la saison en cours
                current_season = self.get_current_season(league_category, league_name)
                if not current_season:
                    print(f"  ⏭ {league_name}: no current season defined, skipping")
                    continue

                if current_season not in seasons_dict:
                    print(f"  ⏭ {league_name}: season {current_season} not in map, skipping")
                    continue

                season_id = seasons_dict[current_season]
                saison = current_season

                print(f"\n🔹 {league_name} — saison {saison} (id: {season_id})")

                # Charge le fichier existant
                safe_saison = saison.replace("/", "-")
                file_path = self.base_path / league_name / f"{safe_saison}.json"
                existing_events = []
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            existing_events = json.load(f)
                        print(f"  📂 {len(existing_events)} événements existants chargés")
                    except Exception as e:
                        print(f"  [WARNING] Could not read {file_path}: {e}")

                rounds_info = self.get_rounds_with_fallback(tournament_id, season_id, league_name, saison)

                if not rounds_info:
                    print(f"  🚨 No rounds info for {league_name} {saison}, skipping.")
                    continue

                updated_events = existing_events.copy()
                has_changes = False

                for round_info in rounds_info:
                    round_number = round_info['round_number']
                    rounds_analysis = self.analyze_existing_events(updated_events)

                    if round_number in rounds_analysis:
                        round_analysis = rounds_analysis[round_number]
                        if round_analysis.get("all_ended", False):
                            print(f"  ⏭ Round {round_number} — all ended, skip")
                            continue
                        else:
                            print(f"  🔄 Round {round_number} — {round_analysis['ended']}/{round_analysis['total']} ended, re-scraping")
                    else:
                        print(f"  🆕 Round {round_number} — new")

                    try:
                        round_events = self.scrape_events_for_round(tournament_id, season_id, round_info)
                        print(f"     📦 {len(round_events)} events — statuts: {set(e.get('description') for e in round_events)}")

                        if round_events:
                            existing_round = [e for e in updated_events if e['round'] == round_number]
                            merged = self.merge_events(existing_round, round_events)
                            updated_events = [e for e in updated_events if e['round'] != round_number]
                            updated_events.extend(merged)
                            has_changes = True

                        time.sleep(random.uniform(1.5, 3.0))

                    except Exception as e:
                        print(f"  [ERROR] Round {round_number}: {e}")
                        time.sleep(random.uniform(5.0, 8.0))
                        continue

                if has_changes:
                    self.save_events_to_json(league_name, saison, updated_events)
                    print(f"  ✅ Saved {len(updated_events)} events for {saison}")
                else:
                    print(f"  📋 No changes for {saison}")

                time.sleep(random.uniform(3.0, 5.0))

            time.sleep(random.uniform(8.0, 12.0))


if __name__ == "__main__":
    scraper = EventIdScraperCurrentSeason()
    scraper.scrape_current_season_only()
    print("\n✅ Scraping saison en cours terminé.")