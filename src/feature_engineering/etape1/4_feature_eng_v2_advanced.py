# src/feature_engineering/etape2/feature_eng_v2_advanced.py

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureBuilderV2:
    """
    🚀 Feature Engineering V2 - Création de features avancées
    
    Input: Dataset après V1 (Elo + forme + H2H)
    Output: Dataset enrichi avec interactions, momentum, efficacité, etc.
    
    Respecte EXACTEMENT les noms de colonnes du dataset V1.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.features_created = []
        self.features_skipped = []
        
        # Seuil de disponibilité minimum pour créer une feature
        self.min_availability = 0.30  # 30% de données non-NaN
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complet de création de features avancées"""
        df = df.copy()
        initial_shape = df.shape
        
        self._log_header("🚀 FEATURE ENGINEERING V2 - ADVANCED")
        self._log(f"Initial shape: {df.shape}")
        self._log(f"Date range: {df['date'].min()} → {df['date'].max()}")
        
        # Analyse disponibilité des données
        self._analyze_availability(df)
        
        # Création des features par catégorie
        df = self._create_temporal_features(df)
        df = self._create_elo_interactions(df)
        df = self._create_momentum_features(df)
        df = self._create_efficiency_features(df)
        df = self._create_balance_features(df)
        df = self._create_pressure_features(df)
        df = self._create_big_chance_features(df)
        df = self._create_h2h_advanced(df)
        df = self._create_stability_features(df)
        df = self._create_xg_features(df)
        df = self._create_completeness_features(df)

        # Nettoyage post-création
        df = self._cleanup_features(df)
        
        self._log_header("✅ FEATURE ENGINEERING V2 COMPLETE")
        self._log(f"Final shape: {df.shape}")
        self._log(f"New features: {df.shape[1] - initial_shape[1]}")
        self._log(f"Features created: {len(self.features_created)}")
        if self.features_skipped:
            self._log(f"Features skipped: {len(self.features_skipped)}")
        
        return df
    
    # ===================================
    # ANALYSE DE DISPONIBILITÉ
    # ===================================
    
    def _analyze_availability(self, df: pd.DataFrame):
        """Analyse quelles stats sont disponibles dans le dataset"""
        self._log_header("📊 ANALYZING DATA AVAILABILITY")
        
        checks = {
            'xG data': 'home_expectedgoals_avg_5',
            'Possession': 'home_ballpossession_avg_5',
            'Duels': 'home_duelwonpercent_avg_5',
            'Big Chances': 'home_bigchancecreated_avg_5',
            'Touches in box': 'home_touchesinoppbox_avg_5',
            'Ball Recovery': 'home_ballrecovery_avg_5',
        }
        
        self.availability = {}
        for name, col in checks.items():
            if col in df.columns:
                avail = (~df[col].isna()).sum() / len(df)
                self.availability[name] = avail
                status = "✅" if avail > self.min_availability else "⚠️"
                self._log(f"{status} {name}: {avail*100:.1f}% available")
            else:
                self.availability[name] = 0.0
                self._log(f"❌ {name}: Column not found")
        
        print()
    
    # ===================================
    # TEMPORAL FEATURES
    # ===================================
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporelles (mois, repos, fatigue)"""
        self._log_header("📅 Creating Temporal Features")
        
        # Convertir date si nécessaire
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'])
        
        # Mois & jour de la semaine
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Partie de saison (début/milieu/fin)
        df = df.sort_values('date')
        df['season_month'] = df.groupby('season')['date'].transform(
            lambda x: (x - x.min()).dt.days // 30
        )
        df['season_stage'] = pd.cut(
            df['season_month'].fillna(5),  # Default = milieu
            bins=[-1, 3, 7, 100],
            labels=[0, 1, 2]  # 0=début, 1=milieu, 2=fin
        ).astype(int)
        
        # Repos (jours depuis dernier match)
        df = df.sort_values(['home_team', 'date'])
        df['days_rest_home'] = df.groupby('home_team')['date'].diff().dt.days.fillna(7).clip(0, 30)
        
        df = df.sort_values(['away_team', 'date'])
        df['days_rest_away'] = df.groupby('away_team')['date'].diff().dt.days.fillna(7).clip(0, 30)
        
        df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
        
        self._add_features([
            'month', 'day_of_week', 'is_weekend', 'season_stage',
            'days_rest_home', 'days_rest_away', 'rest_advantage'
        ])
        
        return df
    
    # ===================================
    # ELO INTERACTIONS
    # ===================================
    
    def _create_elo_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interactions multiplicatives Elo × Forme/Buts"""
        self._log_header("🔄 Creating Elo Interactions")
        
        df['elo_x_form_5'] = df['elo_diff'] * df['diff_form_5']
        df['elo_x_form_10'] = df['elo_diff'] * df['diff_form_10']
        df['elo_x_goals_5'] = df['elo_diff'] * df['diff_goals_scored_avg_5']
        df['elo_x_goals_10'] = df['elo_diff'] * df['diff_goals_scored_avg_10']
        df['elo_x_rest'] = df['elo_diff'] * df['rest_advantage']
        
        # Non-linéarité Elo
        df['elo_diff_squared'] = df['elo_diff'] ** 2
        df['elo_diff_abs'] = df['elo_diff'].abs()
        df['elo_diff_log'] = np.sign(df['elo_diff']) * np.log1p(df['elo_diff_abs'])
        
        self._add_features([
            'elo_x_form_5', 'elo_x_form_10', 'elo_x_goals_5', 'elo_x_goals_10', 'elo_x_rest',
            'elo_diff_squared', 'elo_diff_abs', 'elo_diff_log'
        ])
        
        return df
    
    # ===================================
    # MOMENTUM
    # ===================================
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum = tendance récente (5 matchs) vs historique (10 matchs)"""
        self._log_header("📈 Creating Momentum Features")
        
        # Momentum forme
        df['home_momentum_form'] = df['home_form_5'] - df['home_form_10']
        df['away_momentum_form'] = df['away_form_5'] - df['away_form_10']
        df['diff_momentum_form'] = df['home_momentum_form'] - df['away_momentum_form']
        
        # Momentum buts
        df['home_momentum_goals'] = df['home_goals_scored_avg_5'] - df['home_goals_scored_avg_10']
        df['away_momentum_goals'] = df['away_goals_scored_avg_5'] - df['away_goals_scored_avg_10']
        df['diff_momentum_goals'] = df['home_momentum_goals'] - df['away_momentum_goals']
        
        # Momentum défensif
        df['home_momentum_defense'] = df['home_goals_conceded_avg_10'] - df['home_goals_conceded_avg_5']
        df['away_momentum_defense'] = df['away_goals_conceded_avg_10'] - df['away_goals_conceded_avg_5']
        df['diff_momentum_defense'] = df['home_momentum_defense'] - df['away_momentum_defense']
        
        self._add_features([
            'home_momentum_form', 'away_momentum_form', 'diff_momentum_form',
            'home_momentum_goals', 'away_momentum_goals', 'diff_momentum_goals',
            'home_momentum_defense', 'away_momentum_defense', 'diff_momentum_defense'
        ])
        
        return df
    
    # ===================================
    # EFFICIENCY
    # ===================================
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficacité offensive/défensive (ratios)"""
        self._log_header("⚡ Creating Efficiency Features")
        
        # Shot conversion (buts / tirs cadrés)
        df['home_shot_conversion_5'] = self._safe_div(
            df['home_goals_scored_avg_5'], 
            df['home_shotsongoal_avg_5']
        )
        df['away_shot_conversion_5'] = self._safe_div(
            df['away_goals_scored_avg_5'], 
            df['away_shotsongoal_avg_5']
        )
        df['diff_shot_conversion_5'] = df['home_shot_conversion_5'] - df['away_shot_conversion_5']
        
        # Defense efficiency (buts encaissés / tirs cadrés concédés)
        df['home_defense_efficiency_10'] = self._safe_div(
            df['home_goals_conceded_avg_10'],
            df['home_shotsongoal_conceded_avg_10']
        )
        df['away_defense_efficiency_10'] = self._safe_div(
            df['away_goals_conceded_avg_10'],
            df['away_shotsongoal_conceded_avg_10']
        )
        df['diff_defense_efficiency'] = df['away_defense_efficiency_10'] - df['home_defense_efficiency_10']
        
        self._add_features([
            'home_shot_conversion_5', 'away_shot_conversion_5', 'diff_shot_conversion_5',
            'home_defense_efficiency_10', 'away_defense_efficiency_10', 'diff_defense_efficiency'
        ])
        
        return df
    
    # ===================================
    # BALANCE ATTAQUE/DÉFENSE
    # ===================================
    
    def _create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance offensive vs défensive"""
        self._log_header("⚖️ Creating Balance Features")
        
        # Ratio buts marqués / buts encaissés
        df['home_attack_defense_ratio'] = self._safe_div(
            df['home_goals_scored_avg_10'],
            df['home_goals_conceded_avg_10']
        )
        df['away_attack_defense_ratio'] = self._safe_div(
            df['away_goals_scored_avg_10'],
            df['away_goals_conceded_avg_10']
        )
        df['diff_attack_defense_ratio'] = df['home_attack_defense_ratio'] - df['away_attack_defense_ratio']
        
        # Goal difference directement
        df['home_goal_diff_10'] = df['home_goals_scored_avg_10'] - df['home_goals_conceded_avg_10']
        df['away_goal_diff_10'] = df['away_goals_scored_avg_10'] - df['away_goals_conceded_avg_10']
        df['diff_goal_diff_10'] = df['home_goal_diff_10'] - df['away_goal_diff_10']
        
        self._add_features([
            'home_attack_defense_ratio', 'away_attack_defense_ratio', 'diff_attack_defense_ratio',
            'home_goal_diff_10', 'away_goal_diff_10', 'diff_goal_diff_10'
        ])
        
        return df
    
    # ===================================
    # PRESSION
    # ===================================
    
    def _create_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pression offensive (corners, entrées en zone dangereuse)"""
        self._log_header("🎯 Creating Pressure Features")
        
        # Différence corners
        df['diff_corners_10'] = df['home_cornerkicks_avg_10'] - df['away_cornerkicks_avg_10']
        df['diff_corners_5'] = df['home_cornerkicks_avg_5'] - df['away_cornerkicks_avg_5']
        
        # Différence entrées tiers offensif
        df['diff_finalthird_10'] = df['home_finalthirdentries_avg_10'] - df['away_finalthirdentries_avg_10']
        
        # Composite pressure (corners + tiers offensif normalisés)
        if self.availability.get('Touches in box', 0) > self.min_availability:
            df['diff_touches_oppbox_10'] = df['home_touchesinoppbox_avg_10'] - df['away_touchesinoppbox_avg_10']
            self._add_features(['diff_touches_oppbox_10'])
        
        self._add_features([
            'diff_corners_10', 'diff_corners_5', 'diff_finalthird_10'
        ])
        
        return df
    
    # ===================================
    # BIG CHANCES
    # ===================================
    
    def _create_big_chance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features grandes occasions"""
        self._log_header("🎲 Creating Big Chance Features")
        
        if self.availability.get('Big Chances', 0) > self.min_availability:
            # Différence création
            df['diff_bigchance_created_10'] = (
                df['home_bigchancecreated_avg_10'] - df['away_bigchancecreated_avg_10']
            )
            
            # Conversion big chances (si >0.3 créées en moyenne)
            df['home_bigchance_conversion_10'] = self._safe_div_threshold(
                df['home_bigchancescored_avg_10'],
                df['home_bigchancecreated_avg_10'],
                min_threshold=0.3
            )
            df['away_bigchance_conversion_10'] = self._safe_div_threshold(
                df['away_bigchancescored_avg_10'],
                df['away_bigchancecreated_avg_10'],
                min_threshold=0.3
            )
            df['diff_bigchance_conversion'] = (
                df['home_bigchance_conversion_10'] - df['away_bigchance_conversion_10']
            )
            
            self._add_features([
                'diff_bigchance_created_10', 
                'home_bigchance_conversion_10', 'away_bigchance_conversion_10',
                'diff_bigchance_conversion'
            ])
        else:
            self._skip_feature('Big Chance features (insufficient data)')
        
        return df
    
    # ===================================
    # H2H AVANCÉ
    # ===================================
    
    def _create_h2h_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features H2H avancées"""
        self._log_header("🤝 Creating Advanced H2H Features")
        
        # Win rate H2H
        df['h2h_home_winrate_5'] = self._safe_div(df['h2h_home_wins_5'], df['h2h_matches_5'])
        df['h2h_home_winrate_10'] = self._safe_div(df['h2h_home_wins_10'], df['h2h_matches_10'])
        
        # Momentum H2H (récent vs historique)
        df['h2h_momentum'] = df['h2h_home_winrate_5'] - df['h2h_home_winrate_10']
        
        # Goal difference H2H
        df['h2h_goal_diff_5'] = df['h2h_goals_for_5'] - df['h2h_goals_against_5']
        df['h2h_goal_diff_10'] = df['h2h_goals_for_10'] - df['h2h_goals_against_10']
        
        self._add_features([
            'h2h_home_winrate_5', 'h2h_home_winrate_10', 'h2h_momentum',
            'h2h_goal_diff_5', 'h2h_goal_diff_10'
        ])
        
        return df
    
    # ===================================
    # STABILITY
    # ===================================
    
    def _create_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stabilité/Consistance des performances"""
        self._log_header("📊 Creating Stability Features")
        
        # Volatilité forme (écart 5 vs 10 matchs)
        df['home_form_volatility'] = np.abs(df['home_form_5'] - df['home_form_10'])
        df['away_form_volatility'] = np.abs(df['away_form_5'] - df['away_form_10'])
        df['diff_form_volatility'] = df['home_form_volatility'] - df['away_form_volatility']
        
        # Volatilité buts
        df['home_goals_volatility'] = np.abs(
            df['home_goals_scored_avg_5'] - df['home_goals_scored_avg_10']
        )
        df['away_goals_volatility'] = np.abs(
            df['away_goals_scored_avg_5'] - df['away_goals_scored_avg_10']
        )
        
        self._add_features([
            'home_form_volatility', 'away_form_volatility', 'diff_form_volatility',
            'home_goals_volatility', 'away_goals_volatility'
        ])
        
        return df
    
    # ===================================
    # XG FEATURES (si disponible)
    # ===================================
    
    def _create_xg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features basées sur xG (always create columns, fill if data exists)"""
        self._log_header("🎯 Creating xG Features")

        # Liste des colonnes à créer quoi qu'il arrive
        xg_cols = [
            'home_xg_overperf_10',
            'away_xg_overperf_10',
            'diff_xg_overperf',
            'diff_xg_10',
            'diff_xg_5',
            'xg_momentum'
        ]

        # 1. Créer les colonnes en NaN par défaut
        for col in xg_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 2. Vérifier que les colonnes sources existent
        required_cols_10 = [
            'home_expectedgoals_avg_10', 
            'away_expectedgoals_avg_10',
            'home_goals_scored_avg_10', 
            'away_goals_scored_avg_10'
        ]

        required_cols_5 = [
            'home_expectedgoals_avg_5',
            'away_expectedgoals_avg_5'
        ]

        has_base_10 = all(col in df.columns for col in required_cols_10)
        has_base_5  = all(col in df.columns for col in required_cols_5)

        # 3. Calculer uniquement pour les lignes exploitables
        if has_base_10:
            mask_10 = (
                df['home_expectedgoals_avg_10'].notna() & 
                df['away_expectedgoals_avg_10'].notna()
            )

            df.loc[mask_10, 'home_xg_overperf_10'] = (
                df.loc[mask_10, 'home_goals_scored_avg_10'] -
                df.loc[mask_10, 'home_expectedgoals_avg_10']
            )

            df.loc[mask_10, 'away_xg_overperf_10'] = (
                df.loc[mask_10, 'away_goals_scored_avg_10'] -
                df.loc[mask_10, 'away_expectedgoals_avg_10']
            )

            df.loc[mask_10, 'diff_xg_overperf'] = (
                df.loc[mask_10, 'home_xg_overperf_10'] -
                df.loc[mask_10, 'away_xg_overperf_10']
            )

            df.loc[mask_10, 'diff_xg_10'] = (
                df.loc[mask_10, 'home_expectedgoals_avg_10'] -
                df.loc[mask_10, 'away_expectedgoals_avg_10']
            )

        if has_base_5:
            mask_5 = (
                df['home_expectedgoals_avg_5'].notna() & 
                df['away_expectedgoals_avg_5'].notna()
            )

            df.loc[mask_5, 'diff_xg_5'] = (
                df.loc[mask_5, 'home_expectedgoals_avg_5'] -
                df.loc[mask_5, 'away_expectedgoals_avg_5']
            )

        # Momentum = variation entre court et long terme
        both = df['diff_xg_5'].notna() & df['diff_xg_10'].notna()
        df.loc[both, 'xg_momentum'] = (
            df.loc[both, 'diff_xg_5'] - df.loc[both, 'diff_xg_10']
        )

        # 4. Logging
        if self.verbose:
            filled = df['diff_xg_10'].notna().sum()
            self._log(f"✅ xG features created - filled rows: {filled}")

        # 5. Tracker
        self._add_features(xg_cols)

        return df
    

    def _create_completeness_features(self, df):
        """
        Creates completeness metrics:
        - xG completeness
        - global stats completeness (excluding ID & target columns)
        """

        df = df.copy()

        # -------------------------------------------------------------
        # 1. Identify XG-related columns automatically
        # -------------------------------------------------------------
        xg_cols = [
            col for col in df.columns
            if ("xg" in col.lower()) or ("expectedgoals" in col.lower())
        ]

        # Avoid empty xg list
        if len(xg_cols) == 0:
            # Create missing columns to keep the pipeline aligned
            df["has_xg_stats"] = 0.0
            df["has_xg_stats_flag"] = 0
            print("⚠️ Warning: no XG-related columns detected.")
            return df

        # -------------------------------------------------------------
        # 2. xG completeness score based on NA ratio only on xG columns
        # -------------------------------------------------------------
        df["has_xg_stats"] = df[xg_cols].notna().mean(axis=1)

        # binary flag (adjustable threshold)
        df["has_xg_stats_flag"] = (df["has_xg_stats"] > 0.1).astype(int)
        # Compter et afficher le nombre de matchs avec flag
        n_flagged = df["has_xg_stats_flag"].sum()
        total_matches = len(df)

        print(f" Matches avec xG (flag=1) : {n_flagged} / {total_matches} ({n_flagged/total_matches*100:.1f}%)")



        # -------------------------------------------------------------
        # 3. Global completeness score
        #    → must NOT include target & identifiers
        # -------------------------------------------------------------
        ignore_cols = [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "league",
            "season",
            "result",          # target variable
            "homescore",
            "awayscore",
        ]

        feature_cols = [c for c in df.columns if c not in ignore_cols]

        # ratio of non-null stats
        df["stats_completeness_score"] = df[feature_cols].notna().mean(axis=1)

        return df

    
    # ===================================
    # CLEANUP
    # ===================================
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage léger et safe pour value betting"""
        self._log_header("🧹 Cleaning up features (SAFE MODE)")
        
        # 1. Remplacer inf par NaN (division par 0, etc.)
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self._log(f"⚠️ Found {inf_count} infinite values, replacing with NaN")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Clipping léger UNIQUEMENT sur les ratios aberrants (>10 ou <0)
        ratio_cols = [
            'home_attack_defense_ratio', 'away_attack_defense_ratio',
            'home_shot_conversion_5', 'away_shot_conversion_5',
            'home_defense_efficiency_10', 'away_defense_efficiency_10',
            'home_bigchance_conversion_10', 'away_bigchance_conversion_10',
        ]
        
        clipped_count = 0
        for col in ratio_cols:
            if col in df.columns:
                upper = df[col].quantile(0.99)
                # Clipper seulement si ratio > 10 (délirant)
                if upper > 10:
                    df[col] = df[col].clip(upper=10)
                    clipped_count += 1
                # Clipper valeurs négatives (impossible pour un ratio)
                df[col] = df[col].clip(lower=0)
        
        if clipped_count > 0:
            self._log(f"✂️ Light clipping on {clipped_count} ratio features (>10 or <0)")
        
        # 3. Créer versions log pour features à large range (SANS supprimer l'original)
        log_features = ['elo_diff', 'rest_advantage', 'diff_goal_diff_10']
        log_created = 0
        for col in log_features:
            if col in df.columns and f'{col}_log' not in df.columns:
                df[f'{col}_log'] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                log_created += 1
        
        if log_created > 0:
            self._log(f"📊 Created {log_created} log-transformed features (additive)")
            self.features_created.extend([f'{col}_log' for col in log_features if col in df.columns])
        
        return df
    
    # ===================================
    # UTILS
    # ===================================
    
    def _safe_div(self, num: pd.Series, denom: pd.Series, eps: float = 1e-6) -> pd.Series:
        """Division sécurisée"""
        return num / (denom + eps)
    
    def _safe_div_threshold(
        self, 
        num: pd.Series, 
        denom: pd.Series, 
        min_threshold: float = 0.5,
        eps: float = 1e-6
    ) -> pd.Series:
        """Division sécurisée avec seuil minimum sur dénominateur"""
        result = np.where(
            denom >= min_threshold,
            num / (denom + eps),
            np.nan
        )
        return pd.Series(result, index=num.index)
    
    def _add_features(self, feature_list: list):
        """Ajoute à la liste des features créées"""
        self.features_created.extend(feature_list)
        if self.verbose:
            self._log(f"✅ Created {len(feature_list)} features")
    
    def _skip_feature(self, reason: str):
        """Enregistre une feature skippée"""
        self.features_skipped.append(reason)
        if self.verbose:
            self._log(f"⚠️ Skipped: {reason}")
    
    def _log(self, message: str):
        """Log conditionnel"""
        if self.verbose:
            print(f"  {message}")
    
    def _log_header(self, title: str):
        """Log header"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"{title}")
            print('='*70)
    
    def print_summary(self):
        """Résumé détaillé"""
        print(f"\n{'='*70}")
        print(" FEATURE ENGINEERING V2 SUMMARY")
        print('='*70)
        
        print(f"\n✅ Features Created: {len(self.features_created)}")
        for feat in self.features_created:
            print(f"  • {feat}")
        
        if self.features_skipped:
            print(f"\n⚠️ Features Skipped: {len(self.features_skipped)}")
            for feat in self.features_skipped:
                print(f"  • {feat}")
        
        print(f"\n Data Availability:")
        for name, avail in self.availability.items():
            status = "✅" if avail > self.min_availability else "⚠️"
            print(f"  {status} {name}: {avail*100:.1f}%")
        
        print('='*70 + '\n')


# ===================================
# MAIN EXECUTION
# ===================================
if __name__ == "__main__":
    from pathlib import Path
    
    # Paths
    input_file = "data/clean/prematch/etape1/full_dataset.csv"
    output_file = "data/clean/prematch/etape2/full_dataset_v2.csv"
    
    print("="*70)
    print("🚀 FEATURE ENGINEERING V2 - ADVANCED")
    print("="*70)
    print(f"\n Input: {input_file}")
    print(f" Output: {output_file}\n")
    
    # Load data
    df = pd.read_csv(input_file, parse_dates=['date'])
    
    print(f"Initial shape: {df.shape}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    
    # Build features
    builder = AdvancedFeatureBuilderV2(verbose=True)
    df_enhanced = builder.build_features(df)
    
    # Summary
    builder.print_summary()
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_enhanced.to_csv(output_file, index=False)
    
    print(f"\n Dataset saved: {output_file}")
    print(f"Final shape: {df_enhanced.shape}")
    print(f"\n✅ DONE!")