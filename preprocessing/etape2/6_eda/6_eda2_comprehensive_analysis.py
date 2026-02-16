# src/eda/etape2/eda2_comprehensive_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Pour Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


class ComprehensiveEDA:
    """
    EDA2 approfondi pour préparer le cleaning et le modeling.
    Analyse la qualité, redondance, importance, distributions, temporalité.
    """
    
    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset", target_col: str = 'result'):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.target_col = target_col
        
        # Fix: Assurer date est datetime avec error handling
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Identifier colonnes par type
        self.id_cols = ['event_id', 'date', 'home_team', 'away_team', 'league', 'season']
        self.target_cols = [target_col, 'homescore', 'awayscore']
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in self.numeric_cols if c not in self.id_cols + self.target_cols]
        
        # Résultats stockés
        self.results = {}
        
        print(f"\n{'='*70}")
        print(f" COMPREHENSIVE EDA FOR: {dataset_name}")
        print(f"{'='*70}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Target: {target_col}\n")
    
    def _ensure_year_column(self):
        """Crée colonne year si elle n'existe pas"""
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['date'].dt.year
    
    # ============================================
    # 1️ ANALYSE QUALITÉ DES DONNÉES
    # ============================================
    
    def analyze_data_quality(self):
        """Analyse complète de la qualité des données"""
        print(f"\n{'='*70}")
        print(" DATA QUALITY ANALYSIS")
        print(f"{'='*70}\n")
        
        # NaN par colonne
        nan_summary = self.df[self.feature_cols].isnull().sum().sort_values(ascending=False)
        nan_pct = (nan_summary / len(self.df) * 100).round(2)
        
        nan_df = pd.DataFrame({
            'column': nan_summary.index,
            'nan_count': nan_summary.values,
            'nan_pct': nan_pct.values
        })
        
        # Catégoriser
        severe_nan = nan_df[nan_df['nan_pct'] > 50]
        moderate_nan = nan_df[(nan_df['nan_pct'] > 20) & (nan_df['nan_pct'] <= 50)]
        light_nan = nan_df[(nan_df['nan_pct'] > 0) & (nan_df['nan_pct'] <= 20)]
        
        print(f" NaN Summary:")
        print(f"  🔴 Severe (>50% NaN): {len(severe_nan)} columns")
        print(f"  🟠 Moderate (20-50% NaN): {len(moderate_nan)} columns")
        print(f"  🟡 Light (<20% NaN): {len(light_nan)} columns")
        print(f"  🟢 Complete (0% NaN): {len(nan_df) - len(severe_nan) - len(moderate_nan) - len(light_nan)} columns")
        
        if len(severe_nan) > 0:
            print(f"\n  🔴 Severe NaN columns (top 10):")
            for _, row in severe_nan.head(10).iterrows():
                print(f"     • {row['column']}: {row['nan_pct']:.1f}%")
        
        # NaN par match (rows)
        nan_per_row = self.df[self.feature_cols].isnull().sum(axis=1)
        nan_per_row_pct = (nan_per_row / len(self.feature_cols) * 100)
        
        print(f"\n NaN per Match:")
        print(f"  Mean: {nan_per_row_pct.mean():.1f}%")
        print(f"  Median: {nan_per_row_pct.median():.1f}%")
        print(f"  Max: {nan_per_row_pct.max():.1f}%")
        
        # Matchs problématiques (>70% NaN)
        problematic_matches = nan_per_row_pct > 70
        n_problematic = problematic_matches.sum()
        
        if n_problematic > 0:
            print(f"\n   Problematic matches (>70% NaN): {n_problematic} ({n_problematic/len(self.df)*100:.2f}%)")
            print(f"     → Recommend dropping these matches")
        
        # Analyse temporelle des NaN
        self._analyze_nan_timeline()
        
        # Stocker résultats
        self.results['nan_summary'] = nan_df
        self.results['severe_nan_cols'] = severe_nan['column'].tolist()
        self.results['moderate_nan_cols'] = moderate_nan['column'].tolist()
        self.results['problematic_matches_idx'] = self.df[problematic_matches].index.tolist()
        
        return nan_df
    
    def _analyze_nan_timeline(self):
        """Analyse temporelle des NaN"""
        print(f"\n NaN Timeline Analysis:")
        
        # Fix: Utiliser méthode centralisée
        self._ensure_year_column()
        
        # Par année
        nan_by_year = self.df.groupby('year')[self.feature_cols].apply(
            lambda x: (x.isnull().sum().sum() / (len(x) * len(self.feature_cols)) * 100)
        ).round(2)
        
        print(f"  NaN % by year:")
        for year, pct in nan_by_year.items():
            status = "🔴" if pct > 30 else "🟠" if pct > 15 else "🟢"
            print(f"    {status} {year}: {pct:.1f}%")
        
        # Par ligue
        nan_by_league = self.df.groupby('league')[self.feature_cols].apply(
            lambda x: (x.isnull().sum().sum() / (len(x) * len(self.feature_cols)) * 100)
        ).sort_values(ascending=False).round(2)
        
        print(f"\n  NaN % by league (top 10):")
        for league, pct in nan_by_league.head(10).items():
            status = "🔴" if pct > 30 else "🟠" if pct > 15 else "🟢"
            print(f"    {status} {league}: {pct:.1f}%")
    
    # ============================================
    # 2️ ANALYSE REDONDANCE
    # ============================================
    
    def analyze_redundancy(self):
        """Détecte features redondantes (corrélations, constantes)"""
        print(f"\n{'='*70}")
        print(" REDUNDANCY ANALYSIS")
        print(f"{'='*70}\n")
        
        to_drop = []
        # Colonnes constantes
        constant_cols = []
        for col in self.feature_cols:
            if col in self.df.columns:
                nunique = self.df[col].nunique()
                if nunique <= 1:
                    constant_cols.append(col)
        
        if constant_cols:
            print(f"🔴 Constant columns ({len(constant_cols)}):")
            for col in constant_cols:
                print(f"   • {col}")
        else:
            print(f"✅ No constant columns found")
        
        # Corrélations élevées (>0.95)
        print(f"\n High Correlation Analysis (>0.95):")
        
        # Calculer matrice de corrélation
        numeric_df = self.df[self.feature_cols].select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        
        # Triangle supérieur
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Paires avec corr > 0.95
        high_corr_pairs = []
        for column in upper.columns:
            high_corr = upper[column][upper[column] > 0.95]
            for idx in high_corr.index:
                high_corr_pairs.append((column, idx, high_corr[idx]))
        
        if high_corr_pairs:
            print(f"  Found {len(high_corr_pairs)} highly correlated pairs:")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:20]:
                print(f"    • {col1} ↔ {col2}: {corr:.3f}")
            
            # Suggestions de drop
            to_drop = list(set([pair[1] for pair in high_corr_pairs]))
            print(f"\n   Recommend dropping {len(to_drop)} redundant columns")
        else:
            print(f"  ✅ No highly correlated pairs found")
        
        # Leakage detection (colonnes post-match)
        leakage_candidates = [col for col in self.df.columns if any(
            keyword in col.lower() for keyword in ['score', 'result', 'winner', 'outcome']
        ) and col not in self.target_cols]
        
        if leakage_candidates:
            print(f"\n⚠️ Potential leakage columns ({len(leakage_candidates)}):")
            for col in leakage_candidates:
                print(f"   • {col}")
        
        # Stocker résultats
        self.results['constant_cols'] = constant_cols
        self.results['high_corr_pairs'] = high_corr_pairs
        self.results['redundant_cols'] = to_drop if high_corr_pairs else []
        self.results['leakage_candidates'] = leakage_candidates
        
        return high_corr_pairs
    
    # ============================================
    # 3 ANALYSE IMPORTANCE
    # ============================================
    
    def analyze_feature_importance(self, n_estimators: int = 300, max_samples: int = 10000):
        """Analyse importance via corrélation et Random Forest"""
        print(f"\n{'='*70}")
        print(" FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}\n")
        
        # Préparer données
        df_clean = self.df[self.feature_cols + [self.target_col]].dropna()
        
        if len(df_clean) < 100:
            print("⚠️ Too few complete rows for importance analysis")
            return None
        
        # Limiter échantillon si trop gros
        if len(df_clean) > max_samples:
            df_clean = df_clean.sample(n=max_samples, random_state=42)
            print(f" Sampling {max_samples} rows for faster analysis\n")
        
        # Fix: Avertir si peu de données
        completeness_pct = len(df_clean) / len(self.df) * 100
        print(f" Using {len(df_clean):,} complete rows ({completeness_pct:.1f}% of dataset)\n")
        
        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col]
        
        # 1. Corrélation avec target
        print(" Correlation with target:")
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        print("\n  Top 15 correlated features:")
        for i, (feat, corr) in enumerate(correlations.head(15).items(), 1):
            print(f"    {i:2d}. {feat}: {corr:.4f}")
        
        # 2. Random Forest importance
        print(f"\n Random Forest Importance (n_estimators={n_estimators}):")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("  Training Random Forest...")
        rf.fit(X_train, y_train)
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Accuracy: {rf.score(X_test, y_test):.4f}")
        print("\n  Top 15 important features:")
        for i, row in importances.head(15).iterrows():
            print(f"    {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Identifier features inutiles (importance < 0.001)
        useless_features = importances[importances['importance'] < 0.001]['feature'].tolist()
        if useless_features:
            print(f"\n   {len(useless_features)} features with negligible importance (<0.001)")
            print(f"     → Consider dropping for simplicity")
        
        # Stocker résultats
        self.results['correlations'] = correlations
        self.results['rf_importances'] = importances
        self.results['useless_features'] = useless_features
        
        return importances
    
    # ============================================
    # 4 ANALYSE DISTRIBUTIONS
    # ============================================
    
    def analyze_distributions(self):
        """Analyse distributions, outliers, asymétries"""
        print(f"\n{'='*70}")
        print(" DISTRIBUTION ANALYSIS")
        print(f"{'='*70}\n")
        
        # Statistiques descriptives
        stats = self.df[self.feature_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        
        # Identifier features avec outliers extrêmes
        outlier_features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                p01 = self.df[col].quantile(0.01)
                p99 = self.df[col].quantile(0.99)
                
                # Si range très large (>100x médiane)
                median = self.df[col].median()
                if median != 0 and (p99 - p01) > abs(median) * 100:
                    outlier_features.append((col, p01, p99, p99 - p01))
        
        if outlier_features:
            print(f" Features with extreme outliers ({len(outlier_features)}):")
            for col, p01, p99, range_val in sorted(outlier_features, key=lambda x: x[3], reverse=True)[:10]:
                print(f"   • {col}: range [{p01:.2f}, {p99:.2f}] = {range_val:.2f}")
            print(f"\n    Consider clipping or log transformation")
        
        # Identifier features asymétriques (candidates log transform)
        skewed_features = []
        for col in self.feature_cols:
            if col in self.df.columns and self.df[col].notna().sum() > 0:
                skew = self.df[col].skew()
                if abs(skew) > 2:
                    skewed_features.append((col, skew))
        
        if skewed_features:
            print(f"\n Highly skewed features (|skew| > 2): {len(skewed_features)}")
            for col, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
                print(f"   • {col}: skew = {skew:.2f}")
            print(f"\n    Consider log transformation")
        
        # Features déséquilibrées (>90% de 0 ou 1)
        imbalanced_features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts(normalize=True, dropna=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.9:
                    imbalanced_features.append((col, value_counts.iloc[0]))
        
        if imbalanced_features:
            print(f"\n Imbalanced features (>90% same value): {len(imbalanced_features)}")
            for col, pct in sorted(imbalanced_features, key=lambda x: x[1], reverse=True)[:10]:
                print(f"   • {col}: {pct*100:.1f}% same value")
        
        # Stocker résultats
        self.results['stats'] = stats
        self.results['outlier_features'] = [f[0] for f in outlier_features]
        self.results['skewed_features'] = [f[0] for f in skewed_features]
        self.results['imbalanced_features'] = [f[0] for f in imbalanced_features]
    
    # ============================================
    # 5 ANALYSE TEMPORELLE
    # ============================================
    
    def analyze_temporal_patterns(self):
        """Analyse évolution temporelle des features"""
        print(f"\n{'='*70}")
        print(" TEMPORAL ANALYSIS")
        print(f"{'='*70}\n")
        
        # Fix: Features clés robustes (filtrées si manquantes)
        key_features_candidates = ['elo_diff', 'diff_form_10', 'diff_goals_scored_avg_10']
        key_features = [f for f in key_features_candidates if f in self.df.columns]
        
        if not key_features:
            print("⚠️ No key features available for temporal analysis")
            return
        
        # Fix: Utiliser méthode centralisée
        self._ensure_year_column()
        
        # Évolution de la complétude par année
        completeness_by_year = self.df.groupby('year')[self.feature_cols].apply(
            lambda x: (x.notna().sum().sum() / (len(x) * len(self.feature_cols)) * 100)
        ).round(2)
        
        print(" Data Completeness by Year:")
        for year, pct in completeness_by_year.items():
            status = "🟢" if pct > 80 else "🟠" if pct > 60 else "🔴"
            print(f"  {status} {year}: {pct:.1f}%")
        
        # Stabilité des features (variance par année)
        print(f"\n Feature Stability (variance by year):")
        
        for feat in key_features:
            if feat in self.df.columns:
                variance_by_year = self.df.groupby('year')[feat].var()
                mean_variance = variance_by_year.mean()
                print(f"\n  {feat}:")
                print(f"    Mean variance: {mean_variance:.4f}")
                
                # Détection drift (variance change >50%)
                if variance_by_year.max() > variance_by_year.min() * 1.5:
                    print(f"    ⚠️ Potential drift detected (variance varies by {variance_by_year.max()/variance_by_year.min():.1f}x)")
    
    # ============================================
    # 6 ANALYSE PAR LIGUE
    # ============================================
    
    def analyze_by_league(self):
        """Analyse patterns par ligue"""
        print(f"\n{'='*70}")
        print(" LEAGUE ANALYSIS")
        print(f"{'='*70}\n")
        
        # Complétude par ligue
        completeness_by_league = self.df.groupby('league')[self.feature_cols].apply(
            lambda x: (x.notna().sum().sum() / (len(x) * len(self.feature_cols)) * 100)
        ).sort_values(ascending=False).round(2)
        
        print(" Data Completeness by League:")
        for league, pct in completeness_by_league.head(10).items():
            status = "🟢" if pct > 80 else "🟠" if pct > 60 else "🔴"
            print(f"  {status} {league}: {pct:.1f}%")
        
        # Home advantage par ligue
        print(f"\n Home Advantage by League:")
        home_win_rate = self.df.groupby('league')[self.target_col].apply(lambda x: (x == 1).sum() / len(x) * 100).sort_values(ascending=False)
        
        for league, pct in home_win_rate.head(10).items():
            print(f"  {league}: {pct:.1f}% home wins")
    
    # ============================================
    # 7 RECOMMANDATIONS
    # ============================================
    
    def generate_recommendations(self):
        """Génère recommandations pour le cleaning"""
        print(f"\n{'='*70}")
        print(" CLEANING RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        recommendations = {
            'features_to_drop': [],
            'matches_to_drop': [],
            'imputation_strategy': {},
            'transformations': []
        }
        
        # 1. Features à drop
        drop_reasons = []
        
        if 'severe_nan_cols' in self.results and self.results['severe_nan_cols']:
            drop_reasons.append(('Severe NaN (>50%)', self.results['severe_nan_cols']))
        
        if 'constant_cols' in self.results and self.results['constant_cols']:
            drop_reasons.append(('Constant values', self.results['constant_cols']))
        
        if 'redundant_cols' in self.results and self.results['redundant_cols']:
            drop_reasons.append(('High correlation (>0.95)', self.results['redundant_cols'][:10]))
        
        if 'useless_features' in self.results and self.results['useless_features']:
            drop_reasons.append(('Low importance (<0.001)', self.results['useless_features'][:10]))
        
        if 'leakage_candidates' in self.results and self.results['leakage_candidates']:
            drop_reasons.append(('Potential leakage', self.results['leakage_candidates']))
        
        print(" FEATURES TO DROP:")
        for reason, cols in drop_reasons:
            print(f"\n  {reason} ({len(cols)} features):")
            for col in cols[:5]:
                print(f"    • {col}")
            if len(cols) > 5:
                print(f"    ... and {len(cols) - 5} more")
            recommendations['features_to_drop'].extend(cols)
        
        # Dédupliquer
        recommendations['features_to_drop'] = list(set(recommendations['features_to_drop']))
        
        # 2. Matchs à drop
        if 'problematic_matches_idx' in self.results and self.results['problematic_matches_idx']:
            n_problematic = len(self.results['problematic_matches_idx'])
            print(f"\n MATCHES TO DROP:")
            print(f"  {n_problematic} matches with >70% NaN")
            recommendations['matches_to_drop'] = self.results['problematic_matches_idx']
        
        # 3. Stratégie d'imputation
        print(f"\n IMPUTATION STRATEGY:")
        
        if 'moderate_nan_cols' in self.results and self.results['moderate_nan_cols']:
            moderate_cols = self.results['moderate_nan_cols']
            print(f"  Moderate NaN (20-50%): {len(moderate_cols)} columns")
            print(f"    → Recommend: Median by league OR KNN imputation")
            recommendations['imputation_strategy']['moderate'] = moderate_cols
        
        if 'nan_summary' in self.results:
            light_cols = self.results['nan_summary'][
                (self.results['nan_summary']['nan_pct'] > 0) & 
                (self.results['nan_summary']['nan_pct'] <= 20)
            ]['column'].tolist()
            
            if light_cols:
                print(f"  Light NaN (<20%): {len(light_cols)} columns")
                print(f"    → Recommend: Median by league")
                recommendations['imputation_strategy']['light'] = light_cols
        
        # 4. Transformations
        print(f"\n TRANSFORMATIONS:")
        
        # Fix: Sauvegarder TOUTES les features pour automatisation
        if 'outlier_features' in self.results and self.results['outlier_features']:
            print(f"  Clipping: {len(self.results['outlier_features'])} features with extreme outliers")
            print(f"    → Clip at 1st and 99th percentiles")
            # Sauvegarder TOUTES (pas [:10])
            recommendations['transformations'].append(('clip', self.results['outlier_features']))
            
            # Afficher seulement top 10 en console
            print(f"    Top 10:")
            for feat in self.results['outlier_features'][:10]:
                print(f"      • {feat}")
        
        if 'skewed_features' in self.results and self.results['skewed_features']:
            print(f"  Log transform: {len(self.results['skewed_features'])} highly skewed features")
            # Sauvegarder TOUTES (pas [:10])
            recommendations['transformations'].append(('log', self.results['skewed_features']))
            
            # Afficher seulement top 10 en console
            print(f"    Top 10:")
            for feat in self.results['skewed_features'][:10]:
                print(f"      • {feat}")
        
        # Sauvegarder recommandations
        self.results['recommendations'] = recommendations
        
        return recommendations
    
    # ============================================
    # MAIN PIPELINE
    # ============================================
    
    def run_full_analysis(self, save_results: bool = True):
        """Lance l'analyse complète"""
        print(f"\n Starting comprehensive EDA for: {self.dataset_name}\n")
        
        # 1. Qualité
        self.analyze_data_quality()
        
        # 2. Redondance
        self.analyze_redundancy()
        
        # 3. Importance
        self.analyze_feature_importance()
        
        # 4. Distributions
        self.analyze_distributions()
        
        # 5. Temporel
        self.analyze_temporal_patterns()
        
        # 6. Par ligue
        self.analyze_by_league()
        
        # 7. Recommandations
        recommendations = self.generate_recommendations()
        
        # Sauvegarder résultats
        if save_results:
            self._save_results()
        
        print(f"\n{'='*70}")
        print("✅ COMPREHENSIVE EDA COMPLETE")
        print(f"{'='*70}\n")
        
        return recommendations
    
    def _save_results(self):
        """Sauvegarde les résultats en JSON"""
        output_dir = Path(f"data/eda/etape2/{self.dataset_name.lower().replace(' ', '_')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convertir résultats en JSON-serializable
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                results_json[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                results_json[key] = value.to_dict()
            elif isinstance(value, list):
                results_json[key] = value
            elif isinstance(value, dict):
                results_json[key] = value
        
        output_file = output_dir / "eda_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"\n Results saved: {output_file}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*70)
    print(" COMPREHENSIVE EDA2 - FOOTBALL PREDICTION")
    print("="*70)
    
    # Paths
    no_xg_file = "data/clean/prematch/etape2/split/full_dataset_no_xg.csv"
    with_xg_file = "data/clean/prematch/etape2/split/full_dataset_with_xg.csv"
    
    # ============================================
    # ANALYSE DATASET NO-XG (Principal)
    # ============================================
    print("\n" + "="*70)
    print(" ANALYZING NO-XG DATASET (Primary)")
    print("="*70)
    
    df_no_xg = pd.read_csv(no_xg_file, parse_dates=['date'])
    
    eda_no_xg = ComprehensiveEDA(
        df=df_no_xg,
        dataset_name="NO_XG_Dataset",
        target_col='result'
    )
    
    recommendations_no_xg = eda_no_xg.run_full_analysis(save_results=True)
    
    # ============================================
    # ANALYSE DATASET WITH-XG (Experimental)
    # ============================================
    print("\n\n" + "="*70)
    print(" ANALYZING WITH-XG DATASET (Experimental)")
    print("="*70)
    
    df_with_xg = pd.read_csv(with_xg_file, parse_dates=['date'])
    
    eda_with_xg = ComprehensiveEDA(
        df=df_with_xg,
        dataset_name="WITH_XG_Dataset",
        target_col='result'
    )
    
    recommendations_with_xg = eda_with_xg.run_full_analysis(save_results=True)
    
    # ============================================
    # COMPARAISON NO-XG vs WITH-XG
    # ============================================
    print("\n\n" + "="*70)
    print(" COMPARISON: NO-XG vs WITH-XG")
    print("="*70)
    
    print(f"\n Dataset Sizes:")
    print(f"  NO-XG:  {len(df_no_xg):,} matches, {df_no_xg.shape[1]} features")
    print(f"  WITH-XG: {len(df_with_xg):,} matches, {df_with_xg.shape[1]} features")
    print(f"  Difference: {len(df_no_xg) - len(df_with_xg):,} matches ({(1 - len(df_with_xg)/len(df_no_xg))*100:.1f}% less)")
    
    print(f"\n Features to Drop:")
    print(f"  NO-XG:  {len(recommendations_no_xg['features_to_drop'])} features")
    print(f"  WITH-XG: {len(recommendations_with_xg['features_to_drop'])} features")
    
    print(f"\n Matches to Drop:")
    print(f"  NO-XG:  {len(recommendations_no_xg['matches_to_drop'])} matches")
    print(f"  WITH-XG: {len(recommendations_with_xg['matches_to_drop'])} matches")
    
    # Top features comparaison
    if 'rf_importances' in eda_no_xg.results and 'rf_importances' in eda_with_xg.results:
        print(f"\n Top 10 Most Important Features:")
        
        top_no_xg = eda_no_xg.results['rf_importances'].head(10)['feature'].tolist()
        top_with_xg = eda_with_xg.results['rf_importances'].head(10)['feature'].tolist()
        
        print(f"\n  NO-XG Dataset:")
        for i, feat in enumerate(top_no_xg, 1):
            print(f"    {i:2d}. {feat}")
        
        print(f"\n  WITH-XG Dataset:")
        for i, feat in enumerate(top_with_xg, 1):
            marker = "🆕" if 'xg' in feat.lower() or 'expected' in feat.lower() else "  "
            print(f"    {i:2d}. {marker} {feat}")
        
        # Features communes dans top 10
        common_top = set(top_no_xg) & set(top_with_xg)
        print(f"\n  Common in both top 10: {len(common_top)}")
        for feat in common_top:
            print(f"    • {feat}")
    
    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print("\n\n" + "="*70)
    print(" FINAL SUMMARY & NEXT STEPS")
    print("="*70)
    
    print(f"\n PRIMARY MODEL (NO-XG):")
    print(f"  Dataset: {len(df_no_xg):,} matches")
    print(f"  Features after cleaning: ~{df_no_xg.shape[1] - len(recommendations_no_xg['features_to_drop'])}")
    print(f"  Date range: {df_no_xg['date'].min()} → {df_no_xg['date'].max()}")
    
    print(f"\n EXPERIMENTAL MODEL (WITH-XG):")
    print(f"  Dataset: {len(df_with_xg):,} matches")
    print(f"  Features after cleaning: ~{df_with_xg.shape[1] - len(recommendations_with_xg['features_to_drop'])}")
    print(f"  Date range: {df_with_xg['date'].min()} → {df_with_xg['date'].max()}")
    
    print(f"\n NEXT STEPS:")
    print(f"  1. Review recommendations in:")
    print(f"     • data/eda/etape2/no_xg_dataset/eda_results.json")
    print(f"     • data/eda/etape2/with_xg_dataset/eda_results.json")
    print(f"  2. Implement cleaning based on recommendations")
    print(f"  3. Choose imputation strategy (median by league vs KNN)")
    print(f"  4. Apply transformations (clip, log)")
    print(f"  5. Final validation before modeling")
    
    print(f"\n RECOMMENDATIONS:")
    print(f"  • Focus on NO-XG dataset as primary (more data)")
    print(f"  • Use WITH-XG as experimental comparison")
    print(f"  • Consider ensemble of both models for final predictions")
    
    print("\n" + "="*70)
    print("✅ EDA2 COMPLETE!")
    print("="*70)