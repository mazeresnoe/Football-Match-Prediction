import json
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright
import random

class MatchStatsScraper:
    def __init__(self):
        self.event_ids_path = Path("data/raw/event_ids")
        self.stats_path = Path("data/raw/match_stats")
        self.progress_path = Path("data/raw/scraping_progress")
        
        # Créer les dossiers nécessaires
        os.makedirs(self.stats_path, exist_ok=True)
        os.makedirs(self.progress_path, exist_ok=True)

    def load_progress(self, league, season):
        """Charge le fichier de progression pour reprendre le scraping"""
        safe_season = season.replace("/", "-")
        progress_file = self.progress_path / f"{league}_{safe_season}_progress.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {"scraped_events": [], "failed_events": []}
        return {"scraped_events": [], "failed_events": []}

    def save_progress(self, league, season, progress):
        """Sauvegarde la progression"""
        safe_season = season.replace("/", "-")
        progress_file = self.progress_path / f"{league}_{safe_season}_progress.json"
        
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def scrape_match_statistics(self, event_id, max_retries=3):
        """Scrape les statistiques d'un match spécifique"""
        url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
        match_stats = None
        
        for attempt in range(max_retries):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    page = context.new_page()
                    
                    def handle_response(response):
                        nonlocal match_stats
                        if url in response.url:
                            if response.status == 200:
                                try:
                                    match_stats = response.json()
                                except Exception as e:
                                    print(f"[WARNING] Error parsing JSON for event {event_id}: {e}")
                            elif response.status == 429:  # Rate limit
                                print(f"[RATE_LIMIT] Event {event_id} - waiting...")
                                raise Exception("Rate limited")
                            elif response.status == 404:
                                print(f"[NOT_FOUND] Event {event_id} - no stats available")
                                match_stats = {"error": "not_found"}
                            else:
                                print(f"[ERROR] Event {event_id} - Status {response.status}")
                                raise Exception(f"HTTP {response.status}")
                    
                    page.on("response", handle_response)
                    
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                        page.wait_for_timeout(2000)
                        
                        # Si on a récupéré les stats, on sort de la boucle
                        if match_stats:
                            break
                            
                    except Exception as e:
                        print(f"[ERROR] Event {event_id}, attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) + random.uniform(1, 3)
                            time.sleep(wait_time)
                    finally:
                        browser.close()
                        
            except Exception as e:
                print(f"[CRITICAL] Event {event_id}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 10))
        
        return match_stats

    def parse_statistics(self, stats_data, event_info):
        """Parse seulement les statistiques de la période 'ALL' pour optimiser"""
        if not stats_data or "error" in stats_data:
            return None
        
        # Structure de base du match
        parsed_stats = {
            "event_id": event_info["event_id"],
            "home_team": event_info["home_team"],
            "away_team": event_info["away_team"],
            "date": event_info["date"],
            "round": event_info["round"],
            "tournament_id": event_info["tournament_id"],
            "season_id": event_info["season_id"],
            "description": event_info["description"],
            "homeScore": event_info["homeScore"],
            "awayScore": event_info["awayScore"]
        }
        
        # Chercher seulement la période "ALL"
        periods = stats_data.get("statistics", [])
        all_period = None
        
        for period_data in periods:
            if period_data.get("period") == "ALL":
                all_period = period_data
                break
        
        if not all_period:
            print(f"[WARNING] No 'ALL' period found for event {event_info['event_id']}")
            return parsed_stats  # Retourner au moins les infos de base
        
        # Parser toutes les stats de la période "ALL"
        groups = all_period.get("groups", [])
        
        for group in groups:
            stats_items = group.get("statisticsItems", [])
            
            for stat_item in stats_items:
                # Utiliser la clé standardisée de l'API
                stat_key = stat_item.get("key", "")
                if not stat_key:
                    # Fallback sur le nom nettoyé
                    stat_key = stat_item.get("name", "").lower().replace(" ", "_").replace("%", "pct")
                
                # Priorité aux valeurs numériques
                home_value = stat_item.get("homeValue")
                away_value = stat_item.get("awayValue")
                
                # Fallback sur les strings si pas de valeurs numériques
                if home_value is None:
                    home_value = stat_item.get("home")
                if away_value is None:
                    away_value = stat_item.get("away")
                
                # Colonnes home/away
                home_col = f"home_{stat_key}"
                away_col = f"away_{stat_key}"
                
                parsed_stats[home_col] = home_value
                parsed_stats[away_col] = away_value
                
                # Ajouter les totaux pour les stats fractionnelles (ex: "11/18")
                home_total = stat_item.get("homeTotal")
                away_total = stat_item.get("awayTotal")
                
                if home_total is not None:
                    parsed_stats[f"{home_col}_total"] = home_total
                if away_total is not None:
                    parsed_stats[f"{away_col}_total"] = away_total
        
        return parsed_stats

    def scrape_league_season(self, league, season):
        """Scrape toutes les stats d'une ligue/saison"""
        print(f"\n=== Scraping {league} {season} ===")
        
        # Charger les event_ids
        safe_season = season.replace("/", "-")
        event_ids_file = self.event_ids_path / league / f"{safe_season}.json"
        
        if not event_ids_file.exists():
            print(f"[ERROR] File not found: {event_ids_file}")
            return
        
        with open(event_ids_file, "r", encoding="utf-8") as f:
            events = json.load(f)
        
        # Filtrer seulement les matchs terminés
        ended_events = [e for e in events if e.get("description") == "Ended"]
        print(f"Found {len(ended_events)} ended matches out of {len(events)} total")
        
        if not ended_events:
            print("No ended matches to scrape")
            return
        
        # Charger la progression
        progress = self.load_progress(league, season)
        scraped_events = set(progress["scraped_events"])
        failed_events = set(progress["failed_events"])
        
        # Filtrer les événements à scraper
        events_to_scrape = [e for e in ended_events 
                           if e["event_id"] not in scraped_events 
                           and e["event_id"] not in failed_events]
        
        print(f"Already scraped: {len(scraped_events)}")
        print(f"Failed attempts: {len(failed_events)}")
        print(f"Remaining to scrape: {len(events_to_scrape)}")
        
        if not events_to_scrape:
            print("All matches already processed!")
            return
        
        # Créer l'architecture de dossiers
        league_dir = self.stats_path / league
        os.makedirs(league_dir, exist_ok=True)
        
        # Préparer le fichier de sauvegarde
        output_file = league_dir / f"{safe_season}_stats.json"
        all_match_stats = []
        
        # Charger les stats existantes si le fichier existe
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    all_match_stats = json.load(f)
                print(f"Loaded {len(all_match_stats)} existing match stats")
            except:
                all_match_stats = []
        
        # Scraper chaque match
        for i, event_info in enumerate(events_to_scrape):
            event_id = event_info["event_id"]
            print(f"\n[{i+1}/{len(events_to_scrape)}] Scraping event {event_id}: {event_info['home_team']} vs {event_info['away_team']}")
            
            # Scraper les stats
            stats_data = self.scrape_match_statistics(event_id)
            
            if stats_data:
                if "error" in stats_data:
                    print(f"[SKIP] Event {event_id} - {stats_data['error']}")
                    failed_events.add(event_id)
                else:
                    # Parser les statistiques
                    parsed_stats = self.parse_statistics(stats_data, event_info)
                    if parsed_stats:
                        all_match_stats.append(parsed_stats)
                        scraped_events.add(event_id)
                        print(f"[SUCCESS] Event {event_id} - {len(parsed_stats)} stats fields")
                    else:
                        print(f"[FAILED] Event {event_id} - parsing failed")
                        failed_events.add(event_id)
            else:
                print(f"[FAILED] Event {event_id} - no data received")
                failed_events.add(event_id)
            
            # Sauvegarder la progression tous les 10 matchs
            if (i + 1) % 10 == 0 or i == len(events_to_scrape) - 1:
                # Sauvegarder les stats
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_match_stats, f, indent=2, ensure_ascii=False)
                
                # Sauvegarder la progression
                progress = {
                    "scraped_events": list(scraped_events),
                    "failed_events": list(failed_events),
                    "last_update": datetime.now().isoformat()
                }
                self.save_progress(league, season, progress)
                
                print(f"[CHECKPOINT] Saved {len(all_match_stats)} matches")
            
            # Délai entre les requêtes
            time.sleep(random.uniform(1.5, 3.5))
        
        print(f"\n=== FINISHED {league} {season} ===")
        print(f"Total scraped: {len(scraped_events)}")
        print(f"Total failed: {len(failed_events)}")
        print(f"Success rate: {len(scraped_events)/(len(scraped_events)+len(failed_events))*100:.1f}%")

    def convert_to_csv(self, league, season):
        """Convertit le JSON en CSV pour faciliter l'analyse"""
        safe_season = season.replace("/", "-")
        league_dir = self.stats_path / league
        json_file = league_dir / f"{safe_season}_stats.json"
        csv_file = league_dir / f"{safe_season}_stats.csv"
        
        if not json_file.exists():
            print(f"No JSON file found: {json_file}")
            return
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if data:
            # Optimiser les types de données pour la mémoire
            df = pd.DataFrame(data)
            
            # Conversion intelligente des types
            for col in df.columns:
                if col == 'event_id':
                    df[col] = df[col].astype('int64')
                elif 'date' in col:
                    df[col] = pd.to_datetime(df[col])
                elif col in ['round', 'tournament_id', 'season_id']:
                    df[col] = df[col].astype('int32')  # Plus petit que int64
                elif df[col].dtype == 'object':
                    # Essayer de convertir en numérique si possible
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if not numeric.isna().all():
                        # Si conversion réussie, utiliser float32 au lieu de float64
                        df[col] = numeric.astype('float32')
            
            df.to_csv(csv_file, index=False, encoding="utf-8")
            print(f"Converted to CSV: {csv_file}")
            print(f"Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Sample columns: {list(df.columns)[:10]}")
            
            # Quick data quality check
            print(f"\nData Quality Check:")
            print(f"- Null values: {df.isnull().sum().sum()}")
            print(f"- Duplicate event_ids: {df['event_id'].duplicated().sum()}")
            
            # Vérification spécifique pour la possession (<=100)
            if 'home_ballPossession' in df.columns and 'away_ballPossession' in df.columns:
                possession_sum = df['home_ballPossession'] + df['away_ballPossession']
                weird_possession = possession_sum[possession_sum > 100].count()
                print(f"- Ball possession >100%: {weird_possession}")
        else:
            print("No data to convert")

if __name__ == "__main__":
    scraper = MatchStatsScraper()
    
    base_dir = scraper.event_ids_path
    
    # Parcourir toutes les ligues
    for league_dir in base_dir.iterdir():
        if league_dir.is_dir():
            league = league_dir.name
            print(f"\n=== Processing league: {league} ===")
            
            # Parcourir toutes les saisons disponibles pour cette ligue
            for file in league_dir.glob("*.json"):
                season = file.stem  # ex: "2024-2025"
                season = season.replace("-", "/")  # remettre au format original
                
                try:
                    scraper.scrape_league_season(league, season)
                    scraper.convert_to_csv(league, season)
                except Exception as e:
                    print(f"[ERROR] Failed {league} {season}: {e}")
    
    print("\n✅ Scraping terminé pour toutes les ligues et saisons!")
