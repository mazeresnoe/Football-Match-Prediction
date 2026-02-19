"""
Value Bet Extractor - Production Ready

Ce script extrait les matchs PRÉCIS où parier pour battre les bookmakers.
Basé sur la stratégie "Moderate" qui a prouvé un ROI positif de +1.36%.

Stratégie testée et validée :
- EV minimum : 8%
- Confiance minimum : 50%
- Types de paris : Tous (Home, Draw, Away)
- ROI obtenu : +1.36%
- Win rate : 52.88%
- 711 paris sur le test set

Usage :
    python models/xgboost/core/value_bet_extractor.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths


def calculate_expected_value(model_probs, bookmaker_odds):
    """
    Calcule l'Expected Value pour chaque issue.
    
    EV = (prob_model × odds) - 1
    """
    ev_home = (model_probs[0] * bookmaker_odds[0]) - 1
    ev_draw = (model_probs[1] * bookmaker_odds[1]) - 1
    ev_away = (model_probs[2] * bookmaker_odds[2]) - 1
    return ev_home, ev_draw, ev_away


def extract_value_bets(df, model, features, min_ev=0.08, min_confidence=0.50):
    """
    Extrait les value bets selon la stratégie MODERATE validée.
    
    Stratégie :
    - EV minimum : 8%
    - Confiance minimum : 50%
    - Types : Tous (Home, Draw, Away)
    - ROI attendu : +1.36%
    
    Args:
        df: DataFrame avec les matchs
        model: Modèle ML entraîné
        features: Liste des features du modèle
        min_ev: EV minimum (0.08 = 8%)
        min_confidence: Probabilité minimum du modèle (0.50 = 50%)
    
    Returns:
        DataFrame avec les value bets
    """
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
        
        # Calculer EV pour chaque outcome
        ev_home, ev_draw, ev_away = calculate_expected_value(model_prob, odds)
        
        # Vérifier chaque type de pari
        outcomes = [
            ('Home', ev_home, model_prob[0], odds[0], 1),
            ('Draw', ev_draw, model_prob[1], odds[1], 0),
            ('Away', ev_away, model_prob[2], odds[2], -1)
        ]
        
        for bet_type, ev, model_p, odd, result_value in outcomes:
            # CONDITIONS DE LA STRATÉGIE MODERATE
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
    
    # Filtrer seulement les paris avec résultat connu
    df_with_result = value_bets_df[value_bets_df['won'].notna()].copy()
    
    if len(df_with_result) == 0:
        return {
            'total_bets': len(value_bets_df),
            'has_results': False
        }
    
    # Calcul profits
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
    
    # Stats par outcome
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


def print_summary(stats, value_bets_df):
    """Affiche un résumé des value bets."""
    print(f"\n{'='*70}")
    print(f"  VALUE BETS EXTRAITS - STRATÉGIE MODERATE")
    print(f"{'='*70}")
    
    if not stats:
        print("\nAucun value bet détecté.")
        return
    
    if not stats['has_results']:
        print(f"\n{stats['total_bets']} value bets détectés")
        print("(Résultats non disponibles - matchs futurs ou données manquantes)")
        return
    
    print(f"\nSTATISTIQUES GLOBALES :")
    print(f"   • Nombre de paris    : {stats['total_bets']}")
    print(f"   • Paris gagnés       : {stats['total_won']} ({stats['win_rate']:.1f}%)")
    print(f"   • ROI                : {stats['roi']:+.2f}%")
    print(f"   • Profit total       : {stats['total_profit']:+.0f}€")
    print(f"   • Misé total         : {stats['total_staked']:+.0f}€")
    print(f"   • EV moyen           : {stats['avg_ev']:+.2f}%")
    print(f"   • Cote moyenne       : {stats['avg_odds']:.2f}")
    
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
║           VALUE BET EXTRACTOR - PRODUCTION READY              ║
║                                                               ║
║  Stratégie : MODERATE (EV 8%, Conf 50%)                      ║
║  ROI attendu : +1.36%                                         ║
║  Win rate : 52.88%                                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    with_xg = False
    
    # 1. Charger le modèle
    print(f"\nChargement du modèle...")
    
    # Essayer production d'abord (calibré)
    model_path = SavePaths.get_latest_model('production', with_xg=with_xg)
    
    # Sinon experiments (optimisé)
    if model_path is None:
        print("   Pas de modèle calibré, recherche du modèle optimisé...")
        model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)
    
    if model_path is None:
        print(f"\nAucun modèle trouvé.")
        print("Exécute d'abord 'step2b_optimization.py' ou 'step2c_calibration.py'")
        return
    
    print(f"Modèle : {model_path.name}")
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    print(f"Features : {len(features)}")
    
    # 2. Charger les données
    print(f"\nChargement des données...")
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    print(f"Test set : {len(test_df)} matchs")
    
    # 3. Extraire les value bets
    print(f"\nExtraction des value bets...")
    print(f"Stratégie :")
    print(f"  • EV minimum : 8%")
    print(f"  • Confiance minimum : 50%")
    print(f"  • Types de paris : Tous (Home, Draw, Away)")
    
    value_bets_df = extract_value_bets(
        test_df, 
        model, 
        features,
        min_ev=0.08,
        min_confidence=0.50
    )
    
    # 4. Analyser
    stats = analyze_value_bets(value_bets_df, stake_per_bet=10)
    
    # 5. Afficher résumé
    print_summary(stats, value_bets_df)
    
    # 6. Sauvegarder
    if len(value_bets_df) > 0:
        # Sauvegarder dans production/value_bets/
        output_path = SavePaths.get_result_path(
            category='production/value_bet',
            filename='value_bets_moderate.csv',
            with_xg=with_xg
        )
        value_bets_df.to_csv(output_path, index=False)
        print(f"\nValue bets sauvegardés : {output_path}")
        
        # Sauvegarder les métadonnées
        if stats and stats['has_results']:
            SavePaths.save_metadata(
                category='production/value_bet',
                filename='value_bets_moderate.csv',
                metadata={
                    'strategy': 'Moderate',
                    'min_ev': 0.08,
                    'min_confidence': 0.50,
                    'n_value_bets': stats['total_bets'],
                    'win_rate': stats['win_rate'],
                    'roi': stats['roi'],
                    'expected_roi': 1.36
                },
                with_xg=with_xg
            )
    
    print(f"\n{'='*70}")
    print(f"  EXTRACTION TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()