import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from playwright.sync_api import sync_playwright
from src.utils.mapping.map import ALL_LEAGUES, SEASONS_BY_LEAGUE, MAX_ROUNDS_BY_LEAGUE_AND_SEASON
import random

class EventIdScraper:
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
        """
        Merge events en privilégiant les nouveaux events pour mettre à jour les statuts.
        Garde les anciens events seulement s'ils sont déjà 'Ended' et que le nouveau ne l'est pas.
        """
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
            
            # 🔥 LOGIQUE CORRIGÉE : Toujours prendre le nouveau event SAUF si l'ancien est "Ended" et le nouveau ne l'est pas
            if event_id in existing_by_id:
                old_event = existing_by_id[event_id]
                old_status = old_event.get('description')
                new_status = new_event.get('description')
                
                # Garder l'ancien seulement s'il est Ended et que le nouveau ne l'est pas (protection contre regression)
                if old_status == 'Ended' and new_status != 'Ended':
                    merged_events.append(old_event)
                else:
                    # Sinon, toujours prendre le nouveau (pour update les statuts)
                    merged_events.append(new_event)
            else:
                # Nouvel event jamais vu
                merged_events.append(new_event)
        
        # Ajouter les events qui n'ont pas été re-scrappés
        for existing_event in existing_events:
            if existing_event['event_id'] not in processed_ids:
                merged_events.append(existing_event)
        
        return merged_events

    def analyze_existing_events(self, existing_events):
        """
        Analyse les events existants pour savoir quels rounds sont complètement terminés.
        """
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

    def scrape_all(self):
        for league_category in ALL_LEAGUES:
            print(f"\n🏆 Starting category: {league_category}")
            
            for league_name, league_data in ALL_LEAGUES[league_category].items():
                print(f"\n🔹 Scraping league: {league_name}")
                tournament_id = league_data['tournament_id']
                seasons_dict = SEASONS_BY_LEAGUE.get(league_category, {}).get(league_name, {})

                for saison, season_id in seasons_dict.items():
                    print(f"📅 Season: {saison}")
                    
                    safe_saison = saison.replace("/", "-")
                    file_path = self.base_path / league_name / f"{safe_saison}.json"
                    
                    existing_events = []
                    if file_path.exists():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                existing_events = json.load(f)
                        except Exception as e:
                            print(f"[WARNING] Could not read existing file {file_path}: {e}")
                    
                    rounds_info = self.get_rounds_with_fallback(tournament_id, season_id, league_name, saison)
                    
                    if not rounds_info:
                        print(f"🚨 No rounds info available for {league_name} {saison}, skipping.")
                        continue

                    updated_events = existing_events.copy()
                    has_changes = False

                    for round_info in rounds_info:
                        round_number = round_info['round_number']
                        
                        # 🔄 Recalculer l'analyse AVANT chaque round pour avoir les données à jour
                        rounds_analysis = self.analyze_existing_events(updated_events)
                        
                        # 🔍 Vérifier si ce round existe dans les données existantes
                        if round_number in rounds_analysis:
                            round_analysis = rounds_analysis[round_number]
                            
                            # ⏭️ Skip seulement si TOUS les matchs sont terminés
                            if round_analysis.get("all_ended", False):
                                print(f"⏭️  Skipping round {round_number} (all {round_analysis['ended']}/{round_analysis['total']} ended)")
                                continue
                            else:
                                # 🔄 Forcer le re-scraping si des matchs ne sont pas terminés
                                print(f"🔄 Re-scraping round {round_number}: {round_analysis['ended']}/{round_analysis['total']} ended")
                        else:
                            # 🆕 Round jamais scrappé → on le scrappe
                            print(f"🆕 Scraping new round {round_number}")

                        try:
                            print(f"   🌐 Fetching round {round_number}...")
                            round_events = self.scrape_events_for_round(tournament_id, season_id, round_info)
                            
                            print(f"   📦 Got {len(round_events)} events for round {round_number}")
                            
                            if round_events:
                                # Afficher les statuts récupérés
                                statuses = [e.get('description') for e in round_events]
                                print(f"   📊 Statuses: {set(statuses)}")
                                
                                existing_round_events = [e for e in existing_events if e['round'] == round_number]
                                merged_round_events = self.merge_events(existing_round_events, round_events)
                                
                                updated_events = [e for e in updated_events if e['round'] != round_number]
                                updated_events.extend(merged_round_events)
                                has_changes = True
                            else:
                                print(f"   ⚠️  No events returned from API for round {round_number}")
                            
                            time.sleep(random.uniform(1.5, 3.0))
                            
                        except Exception as e:
                            print(f"[ERROR] Failed to scrape round {round_number}: {e}")
                            time.sleep(random.uniform(5.0, 8.0))
                            continue

                    if has_changes and updated_events != existing_events:
                        self.save_events_to_json(league_name, saison, updated_events)
                        print(f"✅ Updated {len(updated_events)} events for {saison}")
                    else:
                        print(f"📋 No changes for {saison}")

                    time.sleep(random.uniform(3.0, 5.0))

                print(f"🏁 Finished {league_name}, waiting before next league...")
                time.sleep(random.uniform(8.0, 12.0))

            print(f"🎯 Finished category {league_category}, waiting before next...")  
            time.sleep(random.uniform(8.0, 12.0))

if __name__ == "__main__":
    scraper = EventIdScraper()
    scraper.scrape_all()
    print("✅ Scraping terminé.")
