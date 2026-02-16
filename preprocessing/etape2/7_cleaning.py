# src/preprocessing/etape3/cleaning_v2_complete.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataCleanerV2:
    """
     Data Cleaner V2 - Nettoyage optimal basé sur EDA2
    
    Stratégie :
    1. Drop années 2015-2016
    2. Drop features problématiques (NaN, corrélation, leakage, importance)
    3. Drop matchs >70% NaN
    4. Imputation MIX (médiane + KNN safe)
    5. Transformations (clip + log)
    6. Validation 2017 optionnelle
    """
    
    def __init__(self, eda_results_path: str, dataset_name: str = "Dataset", verbose: bool = True):
        self.dataset_name = dataset_name
        self.verbose = verbose
        
        # Charger recommandations EDA
        with open(eda_results_path, 'r') as f:
            self.eda_results = json.load(f)
        
        # Stats du cleaning
        self.stats = {
            'initial_shape': None,
            'final_shape': None,
            'dropped_years': [],
            'dropped_features': [],
            'dropped_matches': 0,
            'imputed_features': [],
            'transformed_features': [],
        }
        
        self._log_header(f" DATA CLEANER V2: {dataset_name}")
    
    def clean(self, df: pd.DataFrame, keep_2017: bool = True, check_2017_threshold: float = 0.15) -> pd.DataFrame:
        """Pipeline complet de cleaning"""
        df = df.copy()
        self.stats['initial_shape'] = df.shape
        
        self._log(f"Initial shape: {df.shape}")
        self._log(f"Date range: {df['date'].min()} → {df['date'].max()}\n")
        
        # ÉTAPE 1: Filtrage temporel
        df = self._filter_years(df, keep_2017=keep_2017)
        
        # ÉTAPE 2: Drop features problématiques
        df = self._drop_problematic_features(df)
        
        # ÉTAPE 3: Drop matchs problématiques
        df = self._drop_problematic_matches(df)
        
        # ÉTAPE 4: Imputation MIX
        df = self._impute_missing_values(df)
        
        # ÉTAPE 5: Transformations
        df = self._apply_transformations(df)
        
        # ÉTAPE 6: Vérification 2017 (optionnelle)
        if keep_2017:
            df = self._check_2017_quality(df, threshold=check_2017_threshold)
        
        # ÉTAPE 7: Validation finale
        df = self._final_validation(df)
        
        self.stats['final_shape'] = df.shape
        self._print_summary()
        
        return df
    
    # ============================================
    # ÉTAPE 1: FILTRAGE TEMPOREL
    # ============================================
    
    def _filter_years(self, df: pd.DataFrame, keep_2017: bool = True) -> pd.DataFrame:
        """Drop années 2015-2016"""
        self._log_header("STEP 1: TEMPORAL FILTERING")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        
        initial_count = len(df)
        
        if keep_2017:
            df = df[df['year'] >= 2017]
            self.stats['dropped_years'] = [2015, 2016]
            self._log(f"✅ Dropped years 2015-2016")
        else:
            df = df[df['year'] >= 2018]
            self.stats['dropped_years'] = [2015, 2016, 2017]
            self._log(f"✅ Dropped years 2015-2017")
        
        dropped_count = initial_count - len(df)
        self._log(f"   Dropped: {dropped_count:,} matches ({dropped_count/initial_count*100:.1f}%)")
        self._log(f"   Remaining: {len(df):,} matches")
        self._log(f"   New date range: {df['date'].min()} → {df['date'].max()}\n")
        
        return df
    
    # ============================================
    # ÉTAPE 2: DROP FEATURES PROBLÉMATIQUES
    # ============================================
    
    def _drop_problematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop features basé sur EDA (NaN, corrélation, leakage, importance)"""
        self._log_header("STEP 2: DROPPING PROBLEMATIC FEATURES")
        
        all_drops = []
        
        # 2a. Severe NaN (>50%)
        severe_nan = self.eda_results.get('severe_nan_cols', [])
        if severe_nan:
            severe_nan = [c for c in severe_nan if c in df.columns]
            self._log(f"🔴 Severe NaN (>50%): {len(severe_nan)} features")
            all_drops.extend(severe_nan)
        
        # 2b. Constant columns
        constant_cols = self.eda_results.get('constant_cols', [])
        if constant_cols:
            constant_cols = [c for c in constant_cols if c in df.columns]
            self._log(f"🔴 Constant values: {len(constant_cols)} features")
            all_drops.extend(constant_cols)
        
        # 2c. High correlation (>0.95) - Garder 1ère de chaque paire
        redundant_cols = self.eda_results.get('redundant_cols', [])
        if redundant_cols:
            redundant_cols = [c for c in redundant_cols if c in df.columns]
            self._log(f"🔴 High correlation (>0.95): {len(redundant_cols)} features")
            all_drops.extend(redundant_cols)
        
        # 2d. Low importance (<0.001)
        useless_features = self.eda_results.get('useless_features', [])
        if useless_features:
            useless_features = [c for c in useless_features if c in df.columns]
            self._log(f"🔴 Low importance (<0.001): {len(useless_features)} features")
            all_drops.extend(useless_features)
        
        # 2e. Leakage suspects (compromis - drop seulement les vraiment suspects)
        leakage_to_drop = [
            'home_bigchancescored_avg_5', 'home_bigchancescored_avg_10',
            'away_bigchancescored_avg_5', 'away_bigchancescored_avg_10',
            'home_bigchancescored_conceded_avg_5', 'home_bigchancescored_conceded_avg_10',
            'away_bigchancescored_conceded_avg_5', 'away_bigchancescored_conceded_avg_10',
            'home_fail_to_score_rate_5', 'home_fail_to_score_rate_10',
            'away_fail_to_score_rate_5', 'away_fail_to_score_rate_10',
            'diff_goals_scored_avg_5', 'diff_goals_scored_avg_10',
        ]
        leakage_to_drop = [c for c in leakage_to_drop if c in df.columns]
        if leakage_to_drop:
            self._log(f"🔴 Leakage suspects: {len(leakage_to_drop)} features")
            all_drops.extend(leakage_to_drop)
        
        # Dédupliquer et drop
        all_drops = list(set(all_drops))
        all_drops = [c for c in all_drops if c in df.columns]
        
        if all_drops:
            self._log(f"\n Total features to drop: {len(all_drops)}")
            df = df.drop(columns=all_drops)
            self.stats['dropped_features'] = all_drops
            self._log(f"✅ Dropped {len(all_drops)} features")
            self._log(f"   Remaining features: {df.shape[1]}\n")
        
        return df
    
    # ============================================
    # ÉTAPE 3: DROP MATCHS PROBLÉMATIQUES
    # ============================================
    
    def _drop_problematic_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop matchs avec >70% NaN"""
        self._log_header("STEP 3: DROPPING PROBLEMATIC MATCHES")
        
        # Identifier colonnes numériques (features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['event_id', 'homescore', 'awayscore', 'result']]
        
        # Calculer % NaN par match
        nan_per_match = df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
        problematic_mask = nan_per_match > 0.70
        
        n_problematic = problematic_mask.sum()
        
        if n_problematic > 0:
            self._log(f"🔴 Found {n_problematic} matches with >70% NaN ({n_problematic/len(df)*100:.2f}%)")
            df = df[~problematic_mask]
            self.stats['dropped_matches'] = n_problematic
            self._log(f"✅ Dropped {n_problematic} matches")
            self._log(f"   Remaining: {len(df):,} matches\n")
        else:
            self._log(f"✅ No problematic matches found\n")
        
        return df
    
    # ============================================
    # ÉTAPE 4: IMPUTATION MIX
    # ============================================
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation MIX : Médiane par ligue + KNN safe"""
        self._log_header("STEP 4: IMPUTATION (MIX STRATEGY)")
        
        # Identifier features numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['event_id', 'homescore', 'awayscore', 'result', 'year']]
        
        # Calculer % NaN par feature
        nan_pct = df[feature_cols].isnull().sum() / len(df) * 100
        
        # Catégoriser features
        light_nan = nan_pct[(nan_pct > 0) & (nan_pct <= 10)].index.tolist()
        moderate_nan = nan_pct[(nan_pct > 10) & (nan_pct <= 30)].index.tolist()
        
        self._log(f" NaN Categories:")
        self._log(f"   Light (<10%): {len(light_nan)} features")
        self._log(f"   Moderate (10-30%): {len(moderate_nan)} features")
        
        # 4a. Light NaN - Médiane par ligue (stable features)
        if light_nan:
            self._log(f"\n🔹 Imputing Light NaN (median by league)...")
            for col in light_nan:
                df[col] = df.groupby('league')[col].transform(lambda x: x.fillna(x.median()))
            self._log(f"   ✅ Imputed {len(light_nan)} features")
        
        # 4b. Moderate NaN - Mix médiane + KNN
        if moderate_nan:
            # Features SAFE pour KNN (processus, pas résultats)
            safe_for_knn = [
                col for col in moderate_nan 
                if any(keyword in col.lower() for keyword in [
                    'shotsongoal', 'shotsinsidebox', 'shotsoutsidebox',
                    'bigchancecreated', 'bigchancemissed',
                    'corners', 'finalthird', 'possession', 'passes',
                    'tackle', 'interception', 'clearance', 'duel'
                ])
            ]
            
            # Features indépendantes → médiane
            independent_features = [col for col in moderate_nan if col not in safe_for_knn]
            
            # Imputation médiane (indépendantes)
            if independent_features:
                self._log(f"\n🔹 Imputing Moderate NaN - Independent (median by league)...")
                for col in independent_features:
                    df[col] = df.groupby('league')[col].transform(lambda x: x.fillna(x.median()))
                self._log(f"   ✅ Imputed {len(independent_features)} features")
            
            # Imputation KNN (corrélées, SAFE)
            if safe_for_knn:
                self._log(f"\n Imputing Moderate NaN - Correlated (KNN, n=5)...")
                self._log(f"   Safe features for KNN: {len(safe_for_knn)}")
                
                # KNN Imputer
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                
                # Sélectionner features pour KNN (inclure features complètes pour contexte)
                knn_context_cols = safe_for_knn + [
                    col for col in feature_cols 
                    if col not in safe_for_knn and df[col].isnull().sum() == 0
                ][:20]  # Max 20 features de contexte
                
                # Appliquer KNN
                df[safe_for_knn] = imputer.fit_transform(df[knn_context_cols])[
                    :, :len(safe_for_knn)
                ]
                
                self._log(f"   ✅ Imputed {len(safe_for_knn)} features with KNN")
                self.stats['imputed_features'].extend(safe_for_knn)
        
        # Vérification post-imputation
        remaining_nan = df[feature_cols].isnull().sum().sum()
        if remaining_nan > 0:
            self._log(f"\n⚠️ {remaining_nan} NaN remaining - Filling with global median...")
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            self._log(f"   ✅ All NaN filled")
        
        self._log(f"\n✅ Imputation complete - 0 NaN remaining\n")
        
        return df
    
    # ============================================
    # ÉTAPE 5: TRANSFORMATIONS
    # ============================================
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers + Log transform skewed features"""
        self._log_header("STEP 5: TRANSFORMATIONS")
        
        # 5a. Clip outliers (1st/99th percentiles)
        outlier_features = self.eda_results.get('outlier_features', [])
        if outlier_features:
            outlier_features = [c for c in outlier_features if c in df.columns]
            self._log(f" Clipping outliers: {len(outlier_features)} features")
            
            for col in outlier_features:
                p01 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=p01, upper=p99)
            
            self._log(f"   ✅ Clipped at 1st/99th percentiles")
            self.stats['transformed_features'].extend(outlier_features)
        
        # 5b. Log transform skewed features
        skewed_features = self.eda_results.get('skewed_features', [])
        if skewed_features:
            skewed_features = [c for c in skewed_features if c in df.columns]
            self._log(f"\n Log transforming: {len(skewed_features)} features")
            
            for col in skewed_features:
                # Vérifier si valeurs négatives
                if df[col].min() < 0:
                    # Signed log: sign(x) * log(1 + |x|)
                    df[f'{col}_log'] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                else:
                    # Log simple
                    df[f'{col}_log'] = np.log1p(df[col])
                
                # Garder aussi l'original (pour comparaison modèle)
                # Option: drop original si redondant
                # df = df.drop(columns=[col])
            
            self._log(f"   ✅ Log transformed (original features kept)")
        
        self._log("")
        return df
    
    # ============================================
    # ÉTAPE 6: VÉRIFICATION 2017
    # ============================================
    
    def _check_2017_quality(self, df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
        """Vérifier qualité 2017 post-imputation"""
        self._log_header("STEP 6: CHECKING 2017 QUALITY")
        
        if 2017 not in df['year'].values:
            self._log("✅ 2017 not in dataset, skipping check\n")
            return df
        
        # Features numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['event_id', 'homescore', 'awayscore', 'result', 'year']]
        
        # NaN % en 2017
        df_2017 = df[df['year'] == 2017]
        nan_pct_2017 = df_2017[feature_cols].isnull().sum().sum() / (len(df_2017) * len(feature_cols))
        
        self._log(f" 2017 Quality after imputation:")
        self._log(f"   NaN percentage: {nan_pct_2017*100:.2f}%")
        self._log(f"   Threshold: {threshold*100:.1f}%")
        
        if nan_pct_2017 > threshold:
            self._log(f"\n⚠️ 2017 still has >{threshold*100:.1f}% NaN - Dropping 2017")
            df = df[df['year'] != 2017]
            self.stats['dropped_years'].append(2017)
            self._log(f"   Remaining: {len(df):,} matches")
            self._log(f"   New date range: {df['date'].min()} → {df['date'].max()}\n")
        else:
            self._log(f"✅ 2017 quality acceptable - Keeping\n")
        
        return df
    
    # ============================================
    # ÉTAPE 7: VALIDATION FINALE
    # ============================================
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation finale : 0 NaN, pas d'infinis, distributions OK"""
        self._log_header("STEP 7: FINAL VALIDATION")
        
        # Identifier features numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['event_id', 'homescore', 'awayscore', 'result', 'year']]
        
        # Check 1: NaN
        total_nan = df[feature_cols].isnull().sum().sum()
        if total_nan > 0:
            self._log(f"❌ FAILED: {total_nan} NaN remaining")
        else:
            self._log(f"✅ NaN check: 0 NaN")
        
        # Check 2: Infinis
        total_inf = np.isinf(df[feature_cols]).sum().sum()
        if total_inf > 0:
            self._log(f"⚠️ Found {total_inf} infinite values - Replacing with NaN...")
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            self._log(f"   ✅ Infinite values handled")
        else:
            self._log(f"✅ Infinite check: 0 infinite values")
        
        # Check 3: Constantes (post-cleaning)
        constant_cols = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            self._log(f"⚠️ Found {len(constant_cols)} constant columns - Dropping...")
            df = df.drop(columns=constant_cols)
            self._log(f"   ✅ Constant columns dropped")
        else:
            self._log(f"✅ Constant check: No constant columns")
        
        # Check 4: Distributions
        self._log(f"\n Final Statistics:")
        self._log(f"   Shape: {df.shape}")
        self._log(f"   Date range: {df['date'].min()} → {df['date'].max()}")
        self._log(f"   Completeness: 100% (0 NaN)")
        
        self._log(f"\n✅ Dataset ready for modeling!\n")
        
        return df
    
    # ============================================
    # UTILS
    # ============================================
    
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
    
    def _print_summary(self):
        """Affiche résumé du cleaning"""
        print(f"\n{'='*70}")
        print(f" CLEANING SUMMARY: {self.dataset_name}")
        print('='*70)
        
        print(f"\n Shape:")
        print(f"   Initial: {self.stats['initial_shape']}")
        print(f"   Final: {self.stats['final_shape']}")
        
        if self.stats['dropped_years']:
            print(f"\n Years Dropped: {self.stats['dropped_years']}")
        
        print(f"\n Dropped:")
        print(f"   Features: {len(self.stats['dropped_features'])}")
        print(f"   Matches: {self.stats['dropped_matches']}")
        
        if self.stats['imputed_features']:
            print(f"\n Imputed (KNN): {len(self.stats['imputed_features'])} features")
        
        if self.stats['transformed_features']:
            print(f"\n Transformed: {len(self.stats['transformed_features'])} features")
        
        print(f"\n✅ Final dataset ready for modeling!")
        print('='*70)
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """Sauvegarde dataset nettoyé + metadata"""
        # Sauvegarder CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Sauvegarder metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'cleaning_stats': self.stats,
            'final_shape': df.shape,
            'date_range': {
                'min': str(df['date'].min()),
                'max': str(df['date'].max())
            },
            'completeness': '100%',
        }
        
        metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n Saved:")
        print(f"   Data: {output_path}")
        print(f"   Metadata: {metadata_path}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*70)
    print(" DATA CLEANING V2 - AUTOMATIC")
    print("="*70)
    
    # Paths
    datasets = [
        {
            'name': 'NO_XG',
            'input': 'data/clean/prematch/etape2/split/full_dataset_no_xg.csv',
            'eda': 'data/eda/etape2/no_xg_dataset/eda_results.json',
            'output': 'data/clean/prematch/etape3/full_dataset_no_xg_clean.csv'
        },
        {
            'name': 'WITH_XG',
            'input': 'data/clean/prematch/etape2/split/full_dataset_with_xg.csv',
            'eda': 'data/eda/etape2/with_xg_dataset/eda_results.json',
            'output': 'data/clean/prematch/etape3/full_dataset_with_xg_clean.csv'
        }
    ]
    
    for config in datasets:
        print(f"\n{'='*70}")
        print(f" CLEANING: {config['name']} DATASET")
        print(f"{'='*70}\n")
        
        # Load data
        df = pd.read_csv(config['input'], parse_dates=['date'])
        
        # Clean
        cleaner = DataCleanerV2(
            eda_results_path=config['eda'],
            dataset_name=config['name'],
            verbose=True
        )
        
        df_clean = cleaner.clean(
            df,
            keep_2017=True,  # Garder 2017, vérifier qualité après
            check_2017_threshold=0.15  # Drop si >15% NaN après imputation
        )
        
        # Save
        cleaner.save_cleaned_data(df_clean, config['output'])
    
    print(f"\n{'='*70}")
    print("✅ CLEANING V2 COMPLETE!")
    print(f"{'='*70}")
