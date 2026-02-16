import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FootballPreMatchBuilder:
    """
    Construit un dataset PRÉ-MATCH avec système ELO, features de forme et H2H.
    Compatible avec colonnes en minuscules, NaN pour stats manquantes, min_matches respecté.
    """
    
    def __init__(self,
                 elo_initial: float = 1500,
                 elo_k_factor: float = 20,
                 home_advantage: float = 100,
                 use_adaptive_k: bool = True,
                 windows: List[int] = [5, 10],
                 weight_recent: bool = True,
                 min_matches: int = 5,
                 use_all_stats: bool = True,
                 verbose: bool = True):
        
        self.elo_initial = elo_initial
        self.elo_k_factor = elo_k_factor
        self.home_advantage = home_advantage
        self.use_adaptive_k = use_adaptive_k
        self.windows = windows
        self.weight_recent = weight_recent
        self.min_matches = min_matches
        
        # Stats tracking
        self.offensive_stats = [
            'expectedgoals', 'shotsongoal', 'totalshotsinsidebox', 'shotsoffgoal',
            'totalshotsoutsidebox', 'bigchancecreated', 'bigchancescored', 'bigchancemissed',
            'touchesinoppbox', 'cornerkicks', 'ballpossession', 'accuratepasses', 'finalthirdentries',
        ]
        
        self.defensive_stats = [
            'goalkeepersaves', 'goalsprevented', 'interceptionwon', 'totalclearance',
            'blockedscoringattempt', 'totaltackle', 'wontacklepercent', 'ballrecovery',
        ]
        
        self.domination_stats = [
            'duelwonpercent', 'groundduelspercentage', 'aerialduelspercentage',
        ]
        
        self.secondary_stats = [
            'yellowcards', 'redcards', 'fouls', 'offsides',
        ]
        
        self.all_stats = (
            self.offensive_stats + self.defensive_stats + 
            self.domination_stats + self.secondary_stats
        ) if use_all_stats else [
            'expectedgoals', 'bigchancecreated', 'bigchancescored', 'bigchancemissed',
            'goalkeepersaves', 'goalsprevented', 'shotsongoal', 'totalshotsinsidebox',
            'touchesinoppbox', 'ballpossession',
        ]
        
        self.elo_ratings = {}
        self.team_history_cache = {}
        self.data_postmatch = None
        self.data_prematch = None
        
        self.stats = {
            'n_teams': 0,
            'n_matches_processed': 0,
            'elo_min': float('inf'),
            'elo_max': float('-inf'),
        }
        
        self.verbose = verbose
        
        if self.verbose:
            self._print_initialization_summary()
    
    def _print_initialization_summary(self):
        print("=" * 70)
        print("  FOOTBALL PRE-MATCH DATASET BUILDER V1 MIN")
        print("=" * 70)
        print(f"\n PARAMÈTRES ELO:")
        print(f"   Initial ELO: {self.elo_initial}")
        print(f"   K-factor: {self.elo_k_factor}")
        print(f"   Home advantage: +{self.home_advantage} ELO")
        print(f"\n PARAMÈTRES FORME:")
        print(f"   Windows: {self.windows}")
        print(f"   Min matches: {self.min_matches}")
        print(f"\n⚽ FEATURES: {len(self.all_stats)} stats tracked\n")
        print("=" * 70 + "\n")
    
    def initialize_elo(self):
        """Initialise tous les ELO à la valeur de départ"""
        home_teams = self.data_postmatch["home_team"].unique()
        away_teams = self.data_postmatch["away_team"].unique()
        all_teams = list(set(home_teams) | set(away_teams))
        
        self.elo_ratings = {team: self.elo_initial for team in all_teams}
        self.stats['n_teams'] = len(all_teams)
        
        if self.verbose:
            print(f"✅ Initialisé {len(all_teams)} équipes à ELO={self.elo_initial}\n")
    
    def calculate_expected_score(self, elo_home: float, elo_away: float, is_home: bool = True) -> float:
        """Calcule la probabilité de victoire selon les ELO"""
        elo_home_adjusted = elo_home + self.home_advantage
        diff = elo_away - elo_home_adjusted
        expected_home = 1 / (1 + 10**(diff / 400))
        return expected_home if is_home else 1 - expected_home
    
    def update_elo(self, team_home: str, team_away: str, score_home: int, score_away: int) -> Tuple[float, float]:
        """Met à jour les ELO après un match"""
        elo_home = self.elo_ratings.get(team_home, self.elo_initial)
        elo_away = self.elo_ratings.get(team_away, self.elo_initial)
        
        expected_home = self.calculate_expected_score(elo_home, elo_away, is_home=True)
        
        if score_home > score_away:
            actual_home, actual_away = 1, 0
        elif score_home == score_away:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0, 1
        
        new_elo_home = elo_home + self.elo_k_factor * (actual_home - expected_home)
        new_elo_away = elo_away + self.elo_k_factor * (actual_away - (1 - expected_home))
        
        self.elo_ratings[team_home] = new_elo_home
        self.elo_ratings[team_away] = new_elo_away
        
        self.stats['elo_min'] = min(self.stats['elo_min'], new_elo_home, new_elo_away)
        self.stats['elo_max'] = max(self.stats['elo_max'], new_elo_home, new_elo_away)
        
        return new_elo_home, new_elo_away
    
    def compute_team_history(self, team_name: str, before_date: pd.Timestamp, n_matches: Optional[int] = None) -> pd.DataFrame:
        """Récupère l'historique normalisé d'une équipe AVANT une date"""
        cache_key = (team_name, before_date.strftime('%Y-%m-%d'))
        
        if cache_key in self.team_history_cache:
            history = self.team_history_cache[cache_key]
            if n_matches and len(history) > n_matches:
                return history.tail(n_matches).reset_index(drop=True)
            return history
        
        mask_home = (self.data_postmatch['home_team'] == team_name) & (self.data_postmatch['date'] < before_date)
        mask_away = (self.data_postmatch['away_team'] == team_name) & (self.data_postmatch['date'] < before_date)
        
        home_matches = self.data_postmatch[mask_home].copy()
        away_matches = self.data_postmatch[mask_away].copy()
        
        base_columns = ['date', 'event_id', 'league', 'season']
        
        # HOME normalization
        home_normalized = home_matches[base_columns].copy()
        home_normalized['opponent'] = home_matches['away_team']
        home_normalized['is_home'] = 1
        home_normalized['goals_scored'] = home_matches['homescore']
        home_normalized['goals_conceded'] = home_matches['awayscore']
        home_normalized['result'] = np.sign(home_matches['homescore'] - home_matches['awayscore'])
        home_normalized['points'] = home_normalized['result'].map({1: 3, 0: 1, -1: 0})
        
        for stat in self.all_stats:
            if f'home_{stat}' in home_matches.columns:
                home_normalized[stat] = home_matches[f'home_{stat}']
            if f'away_{stat}' in home_matches.columns:
                home_normalized[f'{stat}_conceded'] = home_matches[f'away_{stat}']
        
        # AWAY normalization
        away_normalized = away_matches[base_columns].copy()
        away_normalized['opponent'] = away_matches['home_team']
        away_normalized['is_home'] = 0
        away_normalized['goals_scored'] = away_matches['awayscore']
        away_normalized['goals_conceded'] = away_matches['homescore']
        away_normalized['result'] = np.sign(away_matches['awayscore'] - away_matches['homescore'])
        away_normalized['points'] = away_normalized['result'].map({1: 3, 0: 1, -1: 0})
        
        for stat in self.all_stats:
            if f'away_{stat}' in away_matches.columns:
                away_normalized[stat] = away_matches[f'away_{stat}']
            if f'home_{stat}' in away_matches.columns:
                away_normalized[f'{stat}_conceded'] = away_matches[f'home_{stat}']
        
        history = pd.concat([home_normalized, away_normalized], ignore_index=True)
        history = history.sort_values('date').reset_index(drop=True)
        
        self.team_history_cache[cache_key] = history
        
        if n_matches and len(history) > n_matches:
            return history.tail(n_matches).reset_index(drop=True)
        
        return history
    
    def calculate_form_stats(self, history: pd.DataFrame, window: int) -> Dict:
        """Calcule les stats de forme sur une fenêtre"""
        if len(history) < self.min_matches:
            features = {
                f'form_{window}': None,
                f'goals_scored_avg_{window}': None,
                f'goals_conceded_avg_{window}': None,
                f'win_rate_{window}': None,
                f'goal_diff_avg_{window}': None,
                f'clean_sheet_rate_{window}': None,
                f'fail_to_score_rate_{window}': None,
            }
            for stat in self.all_stats:
                features[f'{stat}_avg_{window}'] = None
                features[f'{stat}_conceded_avg_{window}'] = None
            return features
        
        recent_history = history.tail(window)
        n = len(recent_history)
        
        if self.weight_recent:
            weights = np.exp(-np.arange(n-1, -1, -1) / window)
            weights = weights / weights.sum()
        else:
            weights = np.ones(n) / n
        
        features = {
            f'form_{window}': np.average(recent_history['points'], weights=weights),
            f'goals_scored_avg_{window}': np.average(recent_history['goals_scored'], weights=weights),
            f'goals_conceded_avg_{window}': np.average(recent_history['goals_conceded'], weights=weights),
            f'win_rate_{window}': (recent_history['result'] == 1).sum() / n,
        }
        
        for stat in self.all_stats:
            if stat in recent_history.columns and recent_history[stat].notna().any():
                valid_mask = recent_history[stat].notna()
                valid_values = recent_history[stat][valid_mask]
                valid_weights = weights[valid_mask]
                if valid_weights.sum() > 0:
                    valid_weights = valid_weights / valid_weights.sum()
                    features[f'{stat}_avg_{window}'] = np.average(valid_values, weights=valid_weights)
                else:
                    features[f'{stat}_avg_{window}'] = None
            else:
                features[f'{stat}_avg_{window}'] = None
            
            if f'{stat}_conceded' in recent_history.columns and recent_history[f'{stat}_conceded'].notna().any():
                valid_mask = recent_history[f'{stat}_conceded'].notna()
                valid_values = recent_history[f'{stat}_conceded'][valid_mask]
                valid_weights = weights[valid_mask]
                if valid_weights.sum() > 0:
                    valid_weights = valid_weights / valid_weights.sum()
                    features[f'{stat}_conceded_avg_{window}'] = np.average(valid_values, weights=valid_weights)
                else:
                    features[f'{stat}_conceded_avg_{window}'] = None
            else:
                features[f'{stat}_conceded_avg_{window}'] = None
        
        features[f'goal_diff_avg_{window}'] = features[f'goals_scored_avg_{window}'] - features[f'goals_conceded_avg_{window}']
        features[f'clean_sheet_rate_{window}'] = (recent_history['goals_conceded'] == 0).sum() / n
        features[f'fail_to_score_rate_{window}'] = (recent_history['goals_scored'] == 0).sum() / n
        
        return features
    
    def _validate_history(self, history: pd.DataFrame, match_date: pd.Timestamp) -> bool:
        """Valide qu'un historique est utilisable"""
        return True  # On ne skip jamais les matchs, NaN géré dans features
    
    def _calculate_h2h(self, team_home: str, team_away: str, before_date: pd.Timestamp, n_matches: int = 5) -> Dict:
        """Calcule les vrais head-to-head (H2H) entre deux équipes"""
        mask = (
            ((self.data_postmatch["home_team"] == team_home) & (self.data_postmatch["away_team"] == team_away)) |
            ((self.data_postmatch["home_team"] == team_away) & (self.data_postmatch["away_team"] == team_home))
        ) & (self.data_postmatch["date"] < before_date)

        h2h = self.data_postmatch.loc[mask].sort_values("date", ascending=False).head(n_matches)

        if h2h.empty:
            return {
                "h2h_matches": 0,
                "h2h_home_wins": 0,
                "h2h_draws": 0,
                "h2h_away_wins": 0,
                "h2h_goals_for": np.nan,
                "h2h_goals_against": np.nan,
                "h2h_xg_for": np.nan,
                "h2h_xg_against": np.nan
            }

        home_wins = draws = away_wins = 0
        goals_for = goals_against = 0
        xg_for = []
        xg_against = []

        for _, row in h2h.iterrows():
            if row["home_team"] == team_home:
                gf = row["homescore"]
                ga = row["awayscore"]
                xgf = row.get("home_expectedgoals", np.nan)
                xga = row.get("away_expectedgoals", np.nan)
            else:
                gf = row["awayscore"]
                ga = row["homescore"]
                xgf = row.get("away_expectedgoals", np.nan)
                xga = row.get("home_expectedgoals", np.nan)

            goals_for += gf
            goals_against += ga

            if gf > ga:  
                home_wins += 1
            elif gf == ga:
                draws += 1
            else:
                away_wins += 1

            xg_for.append(xgf)
            xg_against.append(xga)

        return {
            "h2h_matches": len(h2h),
            "h2h_home_wins": home_wins,
            "h2h_draws": draws,
            "h2h_away_wins": away_wins,
            "h2h_goals_for": goals_for / len(h2h),
            "h2h_goals_against": goals_against / len(h2h),
            "h2h_xg_for": np.nanmean(xg_for),
            "h2h_xg_against": np.nanmean(xg_against),
        }
    
    def build_prematch_features(self, match_row: pd.Series) -> Dict:
        """Construit TOUTES les features pour UN match"""
        team_home = match_row["home_team"]
        team_away = match_row["away_team"]
        match_date = pd.to_datetime(match_row["date"]) if not isinstance(match_row["date"], pd.Timestamp) else match_row["date"]
        home_score = match_row["homescore"]
        away_score = match_row["awayscore"]
        
        features = {}
        
        # ELO
        elo_home = self.elo_ratings.get(team_home, self.elo_initial)
        elo_away = self.elo_ratings.get(team_away, self.elo_initial)
        features['elo_home'] = elo_home
        features['elo_away'] = elo_away
        features['elo_diff'] = elo_home - elo_away
        features["expected_prob_home"] = self.calculate_expected_score(elo_home, elo_away, is_home=True)
        features["expected_prob_away"] = self.calculate_expected_score(elo_home, elo_away, is_home=False)
        
        # Historique
        history_home = self.compute_team_history(team_home, match_date)
        history_away = self.compute_team_history(team_away, match_date)
        
        # Forme
        for window in self.windows:
            form_home = self.calculate_form_stats(history_home, window)
            form_away = self.calculate_form_stats(history_away, window)
            for k, v in form_home.items():
                features[f'home_{k}'] = v
            for k, v in form_away.items():
                features[f'away_{k}'] = v
        
        # H2H
        for n in [5, 10]:
            h2h = self._calculate_h2h(team_home, team_away, match_date, n)
            for k, v in h2h.items():
                features[f'{k}_{n}'] = v
        
        # Différences simples
        for stat in ['form', 'goals_scored_avg', 'goals_conceded_avg']:
            for window in self.windows:
                home_val = features.get(f'home_{stat}_{window}')
                away_val = features.get(f'away_{stat}_{window}')
                features[f'diff_{stat}_{window}'] = home_val - away_val if home_val is not None and away_val is not None else np.nan
        
        # Colonnes de base
        base_cols = ['date', 'league', 'season', 'event_id', 'home_team', 'away_team', 'homescore', 'awayscore', 'result']
        for col in base_cols:
            features[col] = match_row[col] if col in match_row else np.nan
        
        return features

# --- UTILISATION EXEMPLE ---
if __name__ == "__main__":
    builder = FootballPreMatchBuilder(verbose=True)
    
    for input_file, output_file in [
        ("data/clean/post_match/post_match_clean.csv" , "data/clean/prematch/etape1/full_dataset.csv")
    ]:
        print(f"\n Processing {input_file} ...")
        builder.data_postmatch = pd.read_csv(input_file)
        builder.data_postmatch.columns = builder.data_postmatch.columns.str.lower()
        builder.data_postmatch['date'] = pd.to_datetime(builder.data_postmatch['date'])
        builder.data_postmatch = builder.data_postmatch.sort_values('date').reset_index(drop=True)
        
        builder.initialize_elo()
        prematch_dataset = []
        
        for idx, match_row in builder.data_postmatch.iterrows():
            features = builder.build_prematch_features(match_row)
            prematch_dataset.append(features)
            builder.stats['n_matches_processed'] += 1
            
            builder.update_elo(
                match_row['home_team'],
                match_row['away_team'],
                match_row['homescore'],
                match_row['awayscore']
            )
            
            if (idx + 1) % 100 == 0:
                print(f" {idx + 1}/{len(builder.data_postmatch)} matchs traités")
        
        prematch_df = pd.DataFrame(prematch_dataset)
        first_cols = ["date", "league", "season", "event_id", "home_team", "away_team", "homescore", "awayscore", "result"]
        other_cols = [c for c in prematch_df.columns if c not in first_cols]
        prematch_df = prematch_df[first_cols + other_cols]
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        prematch_df.to_csv(output_file, index=False)
        print(f"\n✅ Dataset pré-match construit : {len(prematch_df):,} lignes")
        print(f" Statistiques ELO :")
        print(f"   Min: {builder.stats['elo_min']:.2f}")
        print(f"   Max: {builder.stats['elo_max']:.2f}")
        print(f"   Matchs traités: {builder.stats['n_matches_processed']}")
        print(f"\n Dataset sauvegardé : {output_file}")
    
        print(prematch_df.columns.tolist())
        print(f"\nShape: {prematch_df.shape}")
        print(f"\nNaN par colonne:\n{prematch_df.isnull().sum().sort_values(ascending=False).head(20)}")