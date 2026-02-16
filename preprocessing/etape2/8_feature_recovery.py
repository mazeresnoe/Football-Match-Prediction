# src/preprocessing/etape3/feature_recovery.py

"""
🔄 Feature Recovery - Récupération des colonnes critiques

Récupère les features droppées par corrélation mais importantes pour le modèle.
"""

import pandas as pd
from pathlib import Path
import numpy as np


def recover_critical_features(
    df_clean: pd.DataFrame,
    df_original: pd.DataFrame,
    dataset_name: str = "Dataset"
) -> pd.DataFrame:
    """
    Récupère les features critiques droppées par corrélation.
    
    Parameters :
    -----------
    df_clean : pd.DataFrame
        Dataset après cleaning
    df_original : pd.DataFrame
        Dataset AVANT cleaning (étape 2)
    dataset_name : str
        "NO_XG" ou "WITH_XG"
        
    Returns :
    -------
    pd.DataFrame
        Dataset avec features critiques récupérées
    """
    
    print(f"\n{'='*70}")
    print(f" RECOVERING CRITICAL FEATURES: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Merger sur event_id pour garder le bon alignement
    merge_key = 'event_id'
    
    # Features CRITIQUES à récupérer (communes aux deux datasets)
    critical_features = {
        'elo_diff': ' Top feature importance (linéaire)',
        'diff_goal_diff_10': ' Top 6-8 en importance',
    }
    
    # Features spécifiques WITH-XG
    if dataset_name == "WITH_XG":
        critical_features.update({
            'away_attack_defense_ratio': ' Info individuelle away',
            'home_form_10': ' Différent de win_rate',
            'away_form_10': ' Différent de win_rate',
        })
    
    # Features optionnelles (forme récente)
    optional_features = {
        'home_shotsongoal_avg_5': ' Forme récente (si disponible)',
        'away_shotsongoal_avg_5': ' Forme récente (si disponible)',
        'home_bigchancecreated_avg_5': ' Momentum offensif',
        'away_bigchancecreated_avg_5': ' Momentum offensif',
    }
    
    # Récupération
    recovered = []
    for feature, reason in {**critical_features, **optional_features}.items():
        if feature in df_original.columns and feature not in df_clean.columns:
            print(f"✅ Recovering: {feature}")
            print(f"   Reason: {reason}")
            
            # Récupérer la colonne depuis l'original
            df_clean[feature] = df_original.loc[df_clean.index, feature]
            recovered.append(feature)
        elif feature not in df_original.columns:
            print(f"⚠️ Skip: {feature} (not in original)")
        else:
            print(f" Already present: {feature}")
    
    print(f"\n Summary:")
    print(f"   Recovered: {len(recovered)} features")
    print(f"   Final shape: {df_clean.shape}")
    
    return df_clean


def smart_correlation_handling(
    df: pd.DataFrame,
    corr_threshold: float = 0.98
) -> pd.DataFrame:
    """
    Alternative : Drop corrélations intelligemment (garde les meilleures).
    
    Stratégie :
    ----------
    - Si corrélation > threshold entre A et B
    - Garde A si A est plus importante OU plus générale
    - Drop B
    
    Exemple :
    --------
    elo_diff (important) vs expected_prob (dérivée) → Garde elo_diff
    _avg_10 (général) vs _avg_5 (spécifique) → Garde _avg_10
    """
    
    print(f"\n{'='*70}")
    print(f" SMART CORRELATION HANDLING (threshold={corr_threshold})")
    print(f"{'='*70}\n")
    
    # Calculer corrélations
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Triangle supérieur
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Features à drop
    to_drop = set()
    
    # Règles de priorité
    priority_keywords = [
        'elo_diff',           # Toujours garder
        'diff_goal_diff',     # Différences importantes
        'diff_form',
        'diff_xg',
        '_avg_10',            # Préférer 10 matchs (plus de données)
        'home_',              # Préférer home/away individuels vs diff
        'away_',
    ]
    
    for column in upper.columns:
        # Trouver colonnes fortement corrélées
        high_corr = upper[column][upper[column] > corr_threshold]
        
        for corr_col in high_corr.index:
            # Décider laquelle garder
            if any(kw in column for kw in priority_keywords):
                # Garder column, drop corr_col
                to_drop.add(corr_col)
                print(f"Drop {corr_col} (corr={upper[column][corr_col]:.3f} with {column})")
            else:
                # Garder corr_col, drop column
                to_drop.add(column)
                print(f"Drop {column} (corr={upper[column][corr_col]:.3f} with {corr_col})")
                break  # Pas besoin de checker les autres
    
    # Drop
    to_drop = [c for c in to_drop if c in df.columns]
    if to_drop:
        print(f"\n Dropping {len(to_drop)} features intelligently")
        df = df.drop(columns=to_drop)
    
    return df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*70)
    print(" FEATURE RECOVERY - CRITICAL FEATURES")
    print("="*70)
    
    # Paths
    datasets = [
        {
            'name': 'NO_XG',
            'clean': 'data/clean/prematch/etape3/full_dataset_no_xg_clean.csv',
            'original': 'data/clean/prematch/etape2/split/full_dataset_no_xg.csv',
            'output': 'data/clean/prematch/etape3/full_dataset_no_xg_clean_v2.csv'
        },
        {
            'name': 'WITH_XG',
            'clean': 'data/clean/prematch/etape3/full_dataset_with_xg_clean.csv',
            'original': 'data/clean/prematch/etape2/split/full_dataset_with_xg.csv',
            'output': 'data/clean/prematch/etape3/full_dataset_with_xg_clean_v2.csv'
        }
    ]
    
    for config in datasets:
        print(f"\n{'='*70}")
        print(f"Processing: {config['name']}")
        print(f"{'='*70}")
        
        # Load
        df_clean = pd.read_csv(config['clean'], parse_dates=['date'])
        df_original = pd.read_csv(config['original'], parse_dates=['date'])
        
        # Aligner les index (même event_id)
        df_clean = df_clean.set_index('event_id')
        df_original = df_original.set_index('event_id')
        
        # Récupérer features critiques
        df_recovered = recover_critical_features(
            df_clean, 
            df_original, 
            dataset_name=config['name']
        )
        
        # Reset index
        df_recovered = df_recovered.reset_index()
        
        # Save
        Path(config['output']).parent.mkdir(parents=True, exist_ok=True)
        df_recovered.to_csv(config['output'], index=False)
        
        print(f"\n Saved: {config['output']}")
        print(f"   Final shape: {df_recovered.shape}")
    
    print(f"\n{'='*70}")
    print("✅ FEATURE RECOVERY COMPLETE!")
    print(f"{'='*70}")