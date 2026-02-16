"""
Fonctions de chargement et préparation des données
"""
import sys
from pathlib import Path
import pandas as pd
from typing import Tuple
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as cfg

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

def load_data(with_xg: bool = False, merge_odds: bool = True) -> pd.DataFrame:
    """Charge le dataset (avec ou sans XG) et merge avec les odds si demandé."""
    print(f"\n{'='*70}\n  CHARGEMENT DES DONNÉES {'AVEC' if with_xg else 'SANS'} XG\n{'='*70}")
    
    data_path = cfg.DATA_WITH_XG if with_xg else cfg.DATA_NO_XG
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {data_path}")
    
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if merge_odds:
        print(" Fusion avec les odds...")
        odds = pd.read_csv(cfg.DATA_ODDS)
        if 'date' in odds.columns:
            odds['date'] = pd.to_datetime(odds['date'], errors='coerce')
        
        # standardiser noms
        for col in ['home_team', 'away_team']:
            df[col+'_clean'] = df[col].str.lower().str.strip().str.replace(' ', '_')
            odds[col+'_clean'] = odds[col].str.lower().str.strip().str.replace(' ', '_')
        
        df = df.merge(
            odds[['date','home_team_clean','away_team_clean','odds_home','odds_draw','odds_away']],
            on=['date','home_team_clean','away_team_clean'],
            how='left'
        )
        df.drop(['home_team_clean','away_team_clean'], axis=1, inplace=True)
        n_with_odds = df['odds_home'].notna().sum()
        print(f" {n_with_odds:,} matchs avec odds ({n_with_odds/len(df)*100:.1f}%)")
    
    print(f" Dataset final : {len(df):,} matchs")
    return df


# ========================================
# SPLIT TEMPORAL TRAIN/CV/TEST
# ========================================

def train_cv_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel basé sur la date."""
    df = df.sort_values('date').reset_index(drop=True)
    n_total = len(df)
    
    train_end_idx = int(n_total * cfg.TRAIN_PCT)
    cv_end_idx = int(n_total * (cfg.TRAIN_PCT + cfg.CV_PCT))
    
    train = df.iloc[:train_end_idx].copy()
    cv = df.iloc[train_end_idx:cv_end_idx].copy()
    test = df.iloc[cv_end_idx:].copy()
    
    print(f" TRAIN : {len(train):,}, CV : {len(cv):,}, TEST : {len(test):,}")
    return train, cv, test


# ========================================
# ODDS → PROBABILITÉS
# ========================================

def odds_to_probs_proportional(odds_home: float, odds_draw: float, odds_away: float):
    """
    Convertit les cotes en probabilités avec la méthode proportionnelle.
    Cette méthode applique plus de marge aux cotes hautes, ce qui est plus réaliste.
    
    Source: http://www.football-data.co.uk/The_Wisdom_of_the_Crowd_updated.pdf
    
    Returns:
        tuple: (prob_home, prob_draw, prob_away)
    """
    # 1. Calculer la marge totale
    marge = ((1/odds_home) + (1/odds_draw) + (1/odds_away)) - 1
    
    # 2. Somme des probabilités inverses (S)
    S = 2 - marge
    
    # 3. Calculer la part de marge pour chaque sélection (P)
    P_home = (1 - 1/odds_home) / S
    P_draw = (1 - 1/odds_draw) / S
    P_away = (1 - 1/odds_away) / S
    
    # 4. Appliquer la marge proportionnellement
    r_home = 1 - ((1 + P_home * marge) - 1/odds_home)
    r_draw = 1 - ((1 + P_draw * marge) - 1/odds_draw)
    r_away = 1 - ((1 + P_away * marge) - 1/odds_away)
    
    # 5. Normaliser (au cas où)
    total = r_home + r_draw + r_away
    
    return r_home/total, r_draw/total, r_away/total


# Garder l'ancienne pour comparaison
def odds_to_probs_simple(odds_home: float, odds_draw: float, odds_away: float):
    """Méthode simple (normalization uniforme)"""
    prob_home_raw = 1 / odds_home
    prob_draw_raw = 1 / odds_draw
    prob_away_raw = 1 / odds_away
    total = prob_home_raw + prob_draw_raw + prob_away_raw
    return prob_home_raw/total, prob_draw_raw/total, prob_away_raw/total


# ========================================
# COMPARAISON DES MÉTHODES
# ========================================

def compare_odds_conversion_methods(odds_home=6.8, odds_draw=4.99, odds_away=1.48):
    """
    Compare les 3 méthodes de conversion des cotes.
    
    Args:
        odds_home, odds_draw, odds_away: Cotes à comparer (défaut: Rennes vs PSG)
    
    Example:
        >>> compare_odds_conversion_methods()
        >>> compare_odds_conversion_methods(2.5, 3.2, 2.8)  # Autre match
    """
    odds = (odds_home, odds_draw, odds_away)
    
    # Méthode 1 : Simple
    p_simple = odds_to_probs_simple(*odds)
    
    # Méthode 2 : Proportionnelle
    p_prop = odds_to_probs_proportional(*odds)
    
    # Méthode 3 : Brute (avec marge)
    marge = sum(1/o for o in odds) - 1
    p_brute = tuple(1/o for o in odds)
    
    print(f"\n{'='*70}")
    print(f"  COMPARAISON DES MÉTHODES DE CONVERSION DES COTES")
    print(f"{'='*70}")
    print(f"\nCotes : {odds[0]} | {odds[1]} | {odds[2]}")
    print(f"Marge : {marge:.2%}")
    print(f"\n{'Méthode':<25} | {'Home':>8} | {'Draw':>8} | {'Away':>8} | {'Total':>8}")
    print(f"{'-'*70}")
    print(f"{'Brute (avec marge)':<25} | {p_brute[0]:>7.2%} | {p_brute[1]:>7.2%} | {p_brute[2]:>7.2%} | {sum(p_brute):>7.2%}")
    print(f"{'Simple (uniforme)':<25} | {p_simple[0]:>7.2%} | {p_simple[1]:>7.2%} | {p_simple[2]:>7.2%} | {sum(p_simple):>7.2%}")
    print(f"{'Proportionnelle ✅':<25} | {p_prop[0]:>7.2%} | {p_prop[1]:>7.2%} | {p_prop[2]:>7.2%} | {sum(p_prop):>7.2%}")
    print(f"{'='*70}\n")
    
    # Différences
    print(f"{'Écart avec Proportionnelle':<25} | {'Home':>8} | {'Draw':>8} | {'Away':>8}")
    print(f"{'-'*70}")
    diff_simple = tuple(p_simple[i] - p_prop[i] for i in range(3))
    print(f"{'Simple vs Prop':<25} | {diff_simple[0]:>+7.2%} | {diff_simple[1]:>+7.2%} | {diff_simple[2]:>+7.2%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test de la fonction
    print("Test des fonctions de conversion :\n")
    compare_odds_conversion_methods()
    compare_odds_conversion_methods(2.5, 3.2, 2.8)  # Match équilibré


# Alias pour compatibilité
odds_to_probs = odds_to_probs_proportional  # ← Utiliser la méthode proportionnelle par défaut