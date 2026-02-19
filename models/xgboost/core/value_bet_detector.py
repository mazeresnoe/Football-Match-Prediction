"""
Value Bet Extractor V2 - Auto-Optimized

Ce script utilise AUTOMATIQUEMENT la meilleure stratégie trouvée par l'optimiseur.

Workflow :
1. Charge la stratégie optimale depuis best_strategy.json
2. Applique cette stratégie au modèle
3. Extrait les matchs où parier
4. Garantit le ROI maximum possible

Usage :
    python models/xgboost/core/value_bet_extractor_v2.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths


def calculate_expected_value(model_probs, bookmaker_odds):
    """Calcule l'Expected Value pour chaque issue."""
    ev_home = (model_probs[0] * bookmaker_odds[0]) - 1
    ev_draw = (model_probs[1] * bookmaker_odds[1]) - 1
    ev_away = (model_probs[2] * bookmaker_odds[2]) - 1
    return ev_home, ev_draw, ev_away


def load_best_strategy(with_xg=False):
    """
    Charge la meilleure stratégie depuis best_strategy.json.
    
    Returns:
        dict: Paramètres de la stratégie optimale
        None: Si aucune stratégie n'est trouvée
    """
    strategy_path = SavePaths.get_result_path(
        category='production/strategy',
        filename='best_strategy.json',
        with_xg=with_xg
    )
    
    if not strategy_path.exists():
        return None
    
    with open(strategy_path, 'r') as f:
        strategy = json.load(f)
    
    return strategy


def extract_value_bets(df, model, features, strategy):
    """
    Extrait les value bets selon la stratégie optimisée.
    
    Args:
        df: DataFrame avec les matchs
        model: Modèle ML entraîné
        features: Liste des features du modèle
        strategy: Dict avec les paramètres (min_ev, min_confidence, bet_types)
    
    Returns:
        DataFrame avec les value bets
    """
    min_ev = strategy['min_ev']
    min_confidence = strategy['min_confidence']
    bet_types = strategy.get('bet_types', None)
    
    # Prédictions
    X = df[features].values
    probs = model.predict_proba(X)
    
    # Filtrer les matchs avec odds
    odds_df = df[['odds_home', 'odds_draw', 'odds_away']].copy()
    mask_odds = odds_df.notna().all(axis=1)
    
    if mask_odds.sum() == 0:
        return pd.DataFrame()
    
    df_filtered = df[mask_odds].copy()
    probs_filtered = probs[mask_odds]
    odds_filtered = odds_df[mask_odds].values
    
    value_bets = []
    
    for idx, (row_idx, row) in enumerate(df_filtered.iterrows()):
        model_prob = probs_filtered[idx]
        odds = odds_filtered[idx]
        
        ev_home, ev_draw, ev_away = calculate_expected_value(model_prob, odds)
        
        outcomes = [
            ('Home', ev_home, model_prob[0], odds[0], 1),
            ('Draw', ev_draw, model_prob[1], odds[1], 0),
            ('Away', ev_away, model_prob[2], odds[2], -1)
        ]
        
        for bet_type, ev, model_p, odd, result_value in outcomes:
            # Filtrer par type si spécifié
            if bet_types and bet_type not in bet_types:
                continue
            
            if ev > min_ev and model_p > min_confidence:
                value_bet = {
                    'date': row['date'],
                    'league': row['league'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'bet_type': bet_type,
                    'model_prob': model_p,
                    'bookmaker_odd': odd,
                    'edge': model_p - (1 / odd),
                    'ev': ev,
                    'ev_pct': ev * 100,
                    'result': row.get(cfg.TARGET_COL, None),
                    'won': row.get(cfg.TARGET_COL, None) == result_value if pd.notna(row.get(cfg.TARGET_COL, None)) else None
                }
                value_bets.append(value_bet)
    
    if value_bets:
        return pd.DataFrame(value_bets).sort_values('ev_pct', ascending=False)
    else:
        return pd.DataFrame()


def analyze_value_bets(value_bets_df, stake_per_bet=10):
    """Analyse les performances des value bets."""
    if len(value_bets_df) == 0:
        return None
    
    df_with_result = value_bets_df[value_bets_df['won'].notna()].copy()
    
    if len(df_with_result) == 0:
        return {
            'total_bets': len(value_bets_df),
            'has_results': False
        }
    
    df_with_result['profit'] = df_with_result.apply(
        lambda row: (row['bookmaker_odd'] - 1) * stake_per_bet if row['won'] 
                   else -stake_per_bet,
        axis=1
    )
    
    total_bets = len(df_with_result)
    total_won = df_with_result['won'].sum()
    win_rate = (total_won / total_bets) * 100
    total_profit = df_with_result['profit'].sum()
    total_staked = total_bets * stake_per_bet
    roi = (total_profit / total_staked) * 100
    
    outcome_stats = df_with_result.groupby('bet_type').agg({
        'won': ['count', 'sum'],
        'profit': 'sum'
    }).round(2)
    
    return {
        'has_results': True,
        'total_bets': total_bets,
        'total_won': int(total_won),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'avg_ev': df_with_result['ev_pct'].mean(),
        'avg_odds': df_with_result['bookmaker_odd'].mean(),
        'outcome_stats': outcome_stats
    }


def print_summary(strategy, stats, value_bets_df):
    """Affiche un résumé des value bets."""
    print(f"\n{'='*70}")
    print(f"  VALUE BETS EXTRAITS - STRATÉGIE OPTIMISÉE")
    print(f"{'='*70}")
    
    print(f"\nSTRATÉGIE UTILISÉE :")
    print(f"   • EV minimum       : {strategy['min_ev']*100:.1f}%")
    print(f"   • Conf minimum     : {strategy['min_confidence']*100:.1f}%")
    print(f"   • Types de paris   : {strategy.get('bet_types', 'Tous')}")
    print(f"   • ROI attendu      : {strategy['expected_roi']:+.2f}%")
    print(f"   • Win rate attendu : {strategy['expected_win_rate']:.1f}%")
    
    if not stats:
        print("\nAucun value bet détecté.")
        return
    
    if not stats['has_results']:
        print(f"\n{stats['total_bets']} value bets détectés")
        print("(Résultats non disponibles - matchs futurs)")
        return
    
    print(f"\nRÉSULTATS RÉELS :")
    print(f"   • Nombre de paris    : {stats['total_bets']}")
    print(f"   • Paris gagnés       : {stats['total_won']} ({stats['win_rate']:.1f}%)")
    print(f"   • ROI                : {stats['roi']:+.2f}%")
    print(f"   • Profit total       : {stats['total_profit']:+.0f}€")
    print(f"   • Misé total         : {stats['total_staked']:+.0f}€")
    print(f"   • EV moyen           : {stats['avg_ev']:+.2f}%")
    print(f"   • Cote moyenne       : {stats['avg_odds']:.2f}")
    
    # Comparaison attendu vs réel
    if stats['roi'] >= strategy['expected_roi'] * 0.9:
        print(f"\n   ROI conforme aux attentes !")
    elif stats['roi'] > 0:
        print(f"\n   ROI positif mais inférieur aux attentes")
    else:
        print(f"\n   ⚠️ ROI négatif (variance ou sur-ajustement)")
    
    print(f"\nPAR TYPE DE PARI :")
    print(stats['outcome_stats'])
    
    print(f"\nTOP 10 VALUE BETS (par EV%) :")
    top_10 = value_bets_df.head(10)
    
    print(f"\n{'Date':<12} {'Match':<40} {'Paris':6} {'Cote':>6} {'EV%':>8} {'Gagné':>6}")
    print(f"{'-'*85}")
    
    for row in top_10.itertuples():
        date_str = row.date.strftime('%Y-%m-%d') if pd.notna(row.date) else 'N/A'
        match = f"{row.home_team[:18]} vs {row.away_team[:18]}"
        won_str = 'V' if row.won else 'X' if pd.notna(row.won) else '?'
        
        print(f"{date_str:<12} {match:<40} {row.bet_type:6} {row.bookmaker_odd:6.2f} "
              f"{row.ev_pct:7.2f}% {won_str:>6}")


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        VALUE BET EXTRACTOR V2 - AUTO-OPTIMIZED               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    with_xg = False
    
    # 1. Charger la stratégie optimale
    print(f"\nChargement de la stratégie optimale...")
    strategy = load_best_strategy(with_xg=with_xg)
    
    if strategy is None:
        print(f"\n⚠️ Aucune stratégie optimisée trouvée !")
        print(f"   Exécute d'abord 'optimize_strategy.py' pour trouver")
        print(f"   la meilleure stratégie.\n")
        
        # Utiliser stratégie par défaut
        print(f"Utilisation de la stratégie par défaut (Moderate)...")
        strategy = {
            'min_ev': 0.08,
            'min_confidence': 0.50,
            'bet_types': None,
            'expected_roi': 1.36,
            'expected_win_rate': 52.88
        }
    else:
        print(f"Stratégie chargée depuis best_strategy.json")
    
    # 2. Charger le modèle
    print(f"\nChargement du modèle...")
    model_path = SavePaths.get_latest_model('production', with_xg=with_xg)
    
    if model_path is None:
        print("   Pas de modèle calibré, recherche du modèle optimisé...")
        model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)
    
    if model_path is None:
        print(f"\nAucun modèle trouvé")
        return
    
    print(f"Modèle : {model_path.name}")
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    print(f"Features : {len(features)}")
    
    # 3. Charger les données
    print(f"\nChargement des données...")
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    print(f"Test set : {len(test_df)} matchs")
    
    # 4. Extraire les value bets
    print(f"\nExtraction des value bets...")
    value_bets_df = extract_value_bets(test_df, model, features, strategy)
    
    # 5. Analyser
    stats = analyze_value_bets(value_bets_df, stake_per_bet=10)
    
    # 6. Afficher résumé
    print_summary(strategy, stats, value_bets_df)
    
    # 7. Sauvegarder
    if len(value_bets_df) > 0:
        output_path = SavePaths.get_result_path(
            category='production/value_bet',
            filename='value_bets_optimized.csv',
            with_xg=with_xg
        )
        value_bets_df.to_csv(output_path, index=False)
        print(f"\nValue bets sauvegardés : {output_path}")
        
        if stats and stats['has_results']:
            SavePaths.save_metadata(
                category='production/value_bet',
                filename='value_bets_optimized.csv',
                metadata={
                    'strategy_min_ev': strategy['min_ev'],
                    'strategy_min_confidence': strategy['min_confidence'],
                    'strategy_bet_types': str(strategy.get('bet_types', 'All')),
                    'expected_roi': strategy['expected_roi'],
                    'actual_roi': stats['roi'],
                    'n_value_bets': stats['total_bets'],
                    'win_rate': stats['win_rate']
                },
                with_xg=with_xg
            )
    
    print(f"\n{'='*70}")
    print(f"  EXTRACTION TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()