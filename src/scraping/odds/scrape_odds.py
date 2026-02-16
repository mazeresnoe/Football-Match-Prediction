"""
🎯 FOOTBALL ODDS SCRAPER - VERSION CORRIGÉE
============================================

Scrape historical odds from Football-Data.co.uk
- Supporte les ligues majeures européennes
- Données historiques 2015-2024
- Bookmakers multiples (Pinnacle, Bet365, etc.)
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FootballOddsScraper:
    """
    Scrape historical odds from Football-Data.co.uk
    """
    
    BASE_URL = "https://www.football-data.co.uk"
    
    # Mapping leagues
    LEAGUES = {
        'Premier League': {'code': 'E0', 'country': 'england'},
        'Championship': {'code': 'E1', 'country': 'england'},
        'La Liga': {'code': 'SP1', 'country': 'spain'},
        'La Liga 2': {'code': 'SP2', 'country': 'spain'},
        'Serie A': {'code': 'I1', 'country': 'italy'},
        'Serie B': {'code': 'I2', 'country': 'italy'},
        'Bundesliga': {'code': 'D1', 'country': 'germany'},
        'Bundesliga 2': {'code': 'D2', 'country': 'germany'},
        'Ligue 1': {'code': 'F1', 'country': 'france'},
        'Ligue 2': {'code': 'F2', 'country': 'france'},
    }
    
    def __init__(self, output_dir: str = "data/odds"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("🎯 FOOTBALL ODDS SCRAPER (CORRECTED VERSION)")
        print("="*70)
        print(f"📁 Output: {self.output_dir}")
        print(f"🏆 Leagues: {len(self.LEAGUES)}")
    
    def scrape_season(
        self, 
        league_code: str, 
        season: str
    ) -> pd.DataFrame:
        """
        Scrape one season for one league.
        
        Parameters:
        -----------
        league_code : str
            League code (e.g., 'E0', 'SP1')
        season : str
            Season in format 'YYYY-YY' (e.g., '2021-22')
        
        Returns:
        --------
        pd.DataFrame or None if failed
        """
        # Build URL - CORRECTION ICI !
        # Format: https://www.football-data.co.uk/mmz4281/2122/E0.csv
        year_short = season.split('-')[0][-2:] + season.split('-')[1]  # '2021-22' → '2122'
        url = f"{self.BASE_URL}/mmz4281/{year_short}/{league_code}.csv"
        
        try:
            # Télécharger avec headers pour éviter blocage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Lire CSV depuis le contenu
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), encoding='latin1', on_bad_lines='skip')
            
            # Vérifier que le DataFrame n'est pas vide
            if len(df) == 0:
                return None
            
            # Add metadata
            df['league_code'] = league_code
            df['season'] = season
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Failed {league_code} {season}: {e}")
            return None
        except Exception as e:
            print(f"  ⚠️ Parse error {league_code} {season}: {e}")
            return None
    
    def scrape_league(
        self, 
        league_name: str, 
        seasons: List[str]
    ) -> pd.DataFrame:
        """
        Scrape multiple seasons for one league.
        """
        print(f"\n{'='*70}")
        print(f"🏆 {league_name}")
        print('='*70)
        
        league_info = self.LEAGUES[league_name]
        league_code = league_info['code']
        
        all_data = []
        
        for season in tqdm(seasons, desc=f"Scraping {league_name}"):
            df = self.scrape_season(league_code, season)
            
            if df is not None:
                all_data.append(df)
            
            time.sleep(1)  # Be nice to server (augmenté à 1s)
        
        if len(all_data) == 0:
            print(f"  ❌ No data scraped for {league_name}")
            return None
        
        # Combine
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"  ✅ Scraped: {len(combined):,} matches")
        print(f"  📅 Seasons: {combined['season'].nunique()}")
        
        return combined
    
    def scrape_all(
        self, 
        seasons: List[str],
        leagues: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all leagues for given seasons.
        
        Parameters:
        -----------
        seasons : List[str]
            List of seasons (e.g., ['2020-21', '2021-22'])
        leagues : List[str], optional
            List of league names. If None, scrape all.
        
        Returns:
        --------
        Dict with league_name → DataFrame
        """
        print(f"\n🎯 SCRAPING {len(seasons)} SEASONS")
        print(f"   Seasons: {seasons[0]} to {seasons[-1]}")
        
        if leagues is None:
            leagues = list(self.LEAGUES.keys())
        
        results = {}
        
        for league_name in leagues:
            df = self.scrape_league(league_name, seasons)
            
            if df is not None:
                results[league_name] = df
                
                # Save individual league
                filename = f"{league_name.lower().replace(' ', '_')}.csv"
                output_path = self.output_dir / filename
                df.to_csv(output_path, index=False)
                print(f"  💾 Saved: {output_path}")
        
        return results
    
    def clean_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize odds data.
        
        Keeps only essential columns + best bookmaker odds.
        """
        print(f"\n{'='*70}")
        print("🧹 CLEANING ODDS DATA")
        print('='*70)
        
        # Essential columns
        essential = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        # Bookmaker columns (priorité: Pinnacle > Bet365 > Max)
        bookmaker_priority = [
            ('Pinnacle', ['PSH', 'PSD', 'PSA']),
            ('Bet365', ['B365H', 'B365D', 'B365A']),
            ('Max', ['MaxH', 'MaxD', 'MaxA']),
            ('Average', ['AvgH', 'AvgD', 'AvgA']),
        ]
        
        # Find available bookmaker
        available = None
        for bookie_name, cols in bookmaker_priority:
            if all(c in df.columns for c in cols):
                available = (bookie_name, cols)
                break
        
        if available is None:
            print("  ⚠️ No bookmaker odds found!")
            print(f"  Available columns: {df.columns.tolist()[:20]}...")
            return None
        
        bookie_name, bookie_cols = available
        
        # Select columns
        keep_cols = essential + bookie_cols + ['league_code', 'season']
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        df_clean = df[keep_cols].copy()
        
        # Rename odds columns
        df_clean = df_clean.rename(columns={
            bookie_cols[0]: 'odds_home',
            bookie_cols[1]: 'odds_draw',
            bookie_cols[2]: 'odds_away',
            'Date': 'date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_score',
            'FTAG': 'away_score',
            'FTR': 'result'
        })
        
        # Convert date (format: DD/MM/YYYY ou DD/MM/YY)
        df_clean['date'] = pd.to_datetime(df_clean['date'], dayfirst=True, errors='coerce')
        
        # Drop rows with missing dates
        df_clean = df_clean.dropna(subset=['date'])
        
        # Drop missing odds
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
        after = len(df_clean)
        
        # Convert odds to float
        for col in ['odds_home', 'odds_draw', 'odds_away']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop invalid odds
        df_clean = df_clean.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
        
        print(f"  📊 Bookmaker: {bookie_name}")
        print(f"  🧹 Dropped {before - after:,} matches with missing odds")
        print(f"  ✅ Final: {len(df_clean):,} matches")
        
        # Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        return df_clean
    
    def merge_with_features(
        self, 
        odds_df: pd.DataFrame, 
        features_df: pd.DataFrame,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Merge odds with your engineered features.
        
        Parameters:
        -----------
        odds_df : pd.DataFrame
            Cleaned odds data
        features_df : pd.DataFrame
            Your feature dataset (from etape3)
        output_path : str, optional
            Where to save merged dataset
        
        Returns:
        --------
        pd.DataFrame with features + odds
        """
        print(f"\n{'='*70}")
        print("🔗 MERGING ODDS WITH FEATURES")
        print('='*70)
        
        # Normalize team names for matching
        def normalize_team(name):
            """Normalise les noms d'équipes pour le matching."""
            if pd.isna(name):
                return ''
            name = str(name).lower().strip()
            # Remplacements communs
            replacements = {
                ' ': '',
                '-': '',
                "'": '',
                '.': '',
                'fc': '',
                'afc': '',
                'utd': 'united',
                'athletic': 'ath',
            }
            for old, new in replacements.items():
                name = name.replace(old, new)
            return name
        
        odds_df['home_team_norm'] = odds_df['home_team'].apply(normalize_team)
        odds_df['away_team_norm'] = odds_df['away_team'].apply(normalize_team)
        
        features_df['home_team_norm'] = features_df['home_team'].apply(normalize_team)
        features_df['away_team_norm'] = features_df['away_team'].apply(normalize_team)
        
        # Convert date to same format
        odds_df['date'] = pd.to_datetime(odds_df['date'])
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # Merge on date + teams
        merged = features_df.merge(
            odds_df[['date', 'home_team_norm', 'away_team_norm', 
                     'odds_home', 'odds_draw', 'odds_away']],
            on=['date', 'home_team_norm', 'away_team_norm'],
            how='left'
        )
        
        # Drop normalization columns
        merged = merged.drop(columns=['home_team_norm', 'away_team_norm'])
        
        # Stats
        before = len(features_df)
        with_odds = merged['odds_home'].notna().sum()
        
        print(f"  📊 Features dataset: {before:,} matches")
        print(f"  🎲 Matched with odds: {with_odds:,} ({with_odds/before*100:.1f}%)")
        print(f"  ❌ Missing odds: {before - with_odds:,} ({(before-with_odds)/before*100:.1f}%)")
        
        # Save if requested
        if output_path:
            merged.to_csv(output_path, index=False)
            print(f"  💾 Saved: {output_path}")
        
        return merged


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # Initialize scraper
    scraper = FootballOddsScraper(output_dir="data/odds/raw")
    
    # Define seasons (2017-18 to 2024-25)
    # NOTE: Limité à 2017+ car tes données SofaScore commencent en 2017
    seasons = []
    for year in range(2017, 2026):
        season = f"{year}-{str(year+1)[-2:]}"
        seasons.append(season)
    
    print(f"\n📅 Seasons to scrape: {len(seasons)}")
    print(f"   Range: {seasons[0]} → {seasons[-1]}")
    
    # Test avec UNE SEULE ligue d'abord
    print("\n🧪 TEST MODE: Scraping Premier League only")
    test_results = scraper.scrape_all(
        seasons=seasons,
        leagues=['Premier League']  # Test avec une seule ligue
    )
    
    if len(test_results) > 0:
        print("\n✅ Test successful! Now scraping all leagues...")
        
        # Scrape all leagues
        results = scraper.scrape_all(
            seasons=seasons,
            leagues=None  # All leagues
        )
        
        # Combine all leagues
        print(f"\n{'='*70}")
        print("📦 COMBINING ALL LEAGUES")
        print('='*70)
        
        all_odds = []
        for league_name, df in results.items():
            all_odds.append(df)
        
        combined_odds = pd.concat(all_odds, ignore_index=True)
        
        print(f"  ✅ Total matches: {len(combined_odds):,}")
        print(f"  🏆 Leagues: {combined_odds['league_code'].nunique()}")
        print(f"  📅 Seasons: {combined_odds['season'].nunique()}")
        
        # Clean
        clean_odds = scraper.clean_odds_data(combined_odds)
        
        if clean_odds is not None:
            # Save combined
            output_path = Path("data/odds/all_odds_clean.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            clean_odds.to_csv(output_path, index=False)
            
            print(f"\n{'='*70}")
            print(f"✅ SCRAPING COMPLETE!")
            print(f"   💾 Saved: {output_path}")
            print(f"   📊 Total: {len(clean_odds):,} matches with odds")
            print('='*70)
            
            # Preview
            print(f"\n📊 PREVIEW:")
            print(clean_odds.head(10))
            
            print(f"\n📈 ODDS STATISTICS:")
            print(clean_odds[['odds_home', 'odds_draw', 'odds_away']].describe())
            
            # Distribution par ligue
            print(f"\n📊 MATCHES BY LEAGUE:")
            print(clean_odds.groupby('league_code').size().sort_values(ascending=False))
    else:
        print("\n❌ Test failed. Check your internet connection or URL format.")