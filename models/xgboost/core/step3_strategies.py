"""
Strategy Optimizer - Grid Search pour maximiser le ROI

Ce script teste TOUTES les combinaisons possibles de paramètres pour trouver
la stratégie avec le ROI le plus élevé.

Grid Search sur :
- EV minimum : 2% à 20% (pas de 1%)
- Confiance minimum : 35% à 65% (pas de 5%)
- Types de paris : Tous, Sans Draw, Home uniquement, Away uniquement

Total : ~19 x 7 x 4 = 532 combinaisons testées

Sauvegarde automatique de la meilleure stratégie dans un fichier JSON.

Usage :
    python models/xgboost/core/optimize_strategy.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
from datetime import datetime
from itertools import product

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


def find_value_bets_strategy(df, model, features, min_ev, min_confidence, bet_types=None):
    """
    Trouve les value bets selon une stratégie donnée.
    
    Args:
        bet_types: Liste des types de paris autorisés ['Home', 'Draw', 'Away']
                  Si None, tous les types sont autorisés
    """
    X = df[features].values
    probs = model.predict_proba(X)
    
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
        
        for bet_type, ev, model_p, odd in [
            ('Home', ev_home, model_prob[0], odds[0]),
            ('Draw', ev_draw, model_prob[1], odds[1]),
            ('Away', ev_away, model_prob[2], odds[2])
        ]:
            # Filtrer par type si spécifié
            if bet_types and bet_type not in bet_types:
                continue
            
            if ev > min_ev and model_p > min_confidence:
                value_bets.append({
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
                    'result': row.get(cfg.TARGET_COL, None)
                })
    
    return pd.DataFrame(value_bets)


def backtest_strategy(value_bets_df, stake_per_bet=10):
    """Backtest d'une stratégie."""
    if len(value_bets_df) == 0:
        return None
    
    df_with_result = value_bets_df[value_bets_df['result'].notna()].copy()
    
    if len(df_with_result) == 0:
        return None
    
    # Déterminer les paris gagnants
    df_with_result['won'] = False
    for idx, row in df_with_result.iterrows():
        result = row['result']
        bet_type = row['bet_type']
        
        if (result == 1 and bet_type == 'Home') or \
           (result == 0 and bet_type == 'Draw') or \
           (result == -1 and bet_type == 'Away'):
            df_with_result.loc[idx, 'won'] = True
    
    # Calcul profits
    df_with_result['profit'] = df_with_result.apply(
        lambda row: (row['bookmaker_odd'] - 1) * stake_per_bet if row['won'] 
                   else -stake_per_bet,
        axis=1
    )
    
    total_bets = len(df_with_result)
    total_won = df_with_result['won'].sum()
    win_rate = total_won / total_bets if total_bets > 0 else 0
    total_profit = df_with_result['profit'].sum()
    total_staked = total_bets * stake_per_bet
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    
    return {
        'total_bets': total_bets,
        'total_won': total_won,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'avg_ev': df_with_result['ev_pct'].mean(),
        'avg_odds': df_with_result['bookmaker_odd'].mean(),
    }


def optimize_strategy(test_df, model, features, min_bets_per_year=100):
    """
    Optimisation par Grid Search pour trouver la meilleure stratégie.
    
    Deux stratégies :
    1. ROI maximum absolu (sans contrainte)
    2. ROI maximum avec minimum de paris par an
    
    Args:
        test_df: DataFrame de test
        model: Modèle entraîné
        features: Features du modèle
        min_bets_per_year: Nombre minimum de paris par an requis (défaut: 100)
    
    Returns:
        best_strategy_pure: Dict avec les meilleurs paramètres (sans contrainte)
        best_strategy_volume: Dict avec les meilleurs paramètres (avec contrainte volume)
        all_results: DataFrame avec tous les résultats testés
    """
    print(f"\n{'='*70}")
    print(f"  OPTIMISATION DE STRATÉGIE - GRID SEARCH")
    print(f"{'='*70}\n")
    
    # Calculer le nombre d'années dans le test set
    test_df_with_odds = test_df[test_df[['odds_home', 'odds_draw', 'odds_away']].notna().all(axis=1)].copy()
    
    if 'date' in test_df_with_odds.columns:
        date_range = (test_df_with_odds['date'].max() - test_df_with_odds['date'].min()).days / 365.25
        n_years = max(1, date_range)
    else:
        # Estimation basée sur le nombre de matchs (38 matchs/équipe/saison pour les ligues)
        n_years = len(test_df_with_odds) / 2000  # ~2000 matchs/an pour toutes les ligues
    
    min_total_bets = int(min_bets_per_year * n_years)
    
    print(f"Paramètres de sélectivité :")
    print(f"  • Période test set : {n_years:.2f} ans")
    print(f"  • Matchs disponibles : {len(test_df_with_odds)}")
    print(f"  • Minimum requis : {min_bets_per_year} paris/an")
    print(f"  • Minimum total : {min_total_bets} paris pour {n_years:.2f} ans")
    print()
    
    # Définir la grille de recherche
    ev_range = np.arange(0.02, 0.21, 0.01)  # 2% à 20%, pas de 1%
    conf_range = np.arange(0.35, 0.66, 0.05)  # 35% à 65%, pas de 5%
    bet_types_options = [
        None,  # Tous
        ['Home', 'Away'],  # Sans Draw
        ['Home'],  # Home uniquement
        ['Away']   # Away uniquement
    ]
    
    total_combinations = len(ev_range) * len(conf_range) * len(bet_types_options)
    print(f"Grille de recherche :")
    print(f"  • EV minimum : {len(ev_range)} valeurs (2% à 20%)")
    print(f"  • Confiance minimum : {len(conf_range)} valeurs (35% à 65%)")
    print(f"  • Types de paris : {len(bet_types_options)} options")
    print(f"  • TOTAL : {total_combinations} combinaisons à tester\n")
    
    results = []
    best_roi_pure = -float('inf')
    best_strategy_pure = None
    
    best_roi_volume = -float('inf')
    best_strategy_volume = None
    
    # Grid Search
    for i, (ev, conf, bet_types) in enumerate(product(ev_range, conf_range, bet_types_options)):
        if (i + 1) % 50 == 0:
            print(f"Progression : {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        
        # Trouver les value bets
        value_bets = find_value_bets_strategy(
            test_df, model, features,
            min_ev=ev,
            min_confidence=conf,
            bet_types=bet_types
        )
        
        if len(value_bets) == 0:
            continue
        
        # Backtest
        backtest = backtest_strategy(value_bets, stake_per_bet=10)
        
        if backtest:
            bet_types_str = str(bet_types) if bet_types else 'All'
            
            # Calculer paris par an
            bets_per_year = backtest['total_bets'] / n_years
            
            results.append({
                'min_ev': ev * 100,
                'min_confidence': conf * 100,
                'bet_types': bet_types_str,
                'total_bets': backtest['total_bets'],
                'bets_per_year': bets_per_year,
                'win_rate': backtest['win_rate'] * 100,
                'roi': backtest['roi'],
                'profit': backtest['total_profit'],
                'avg_ev': backtest['avg_ev'],
                'avg_odds': backtest['avg_odds'],
                'meets_volume_requirement': bets_per_year >= min_bets_per_year
            })
            
            # STRATÉGIE 1 : Meilleur ROI PURE (sans contrainte)
            if backtest['roi'] > best_roi_pure and backtest['total_bets'] >= 10:  # Minimum 10 paris pour éviter variance
                best_roi_pure = backtest['roi']
                best_strategy_pure = {
                    'name': 'Best ROI Pure',
                    'min_ev': ev,
                    'min_confidence': conf,
                    'bet_types': bet_types,
                    'expected_roi': backtest['roi'],
                    'expected_win_rate': backtest['win_rate'] * 100,
                    'expected_bets': backtest['total_bets'],
                    'expected_bets_per_year': bets_per_year,
                    'avg_ev': backtest['avg_ev'],
                    'avg_odds': backtest['avg_odds']
                }
            
            # STRATÉGIE 2 : Meilleur ROI AVEC contrainte volume (≥100 paris/an)
            if bets_per_year >= min_bets_per_year and backtest['roi'] > best_roi_volume:
                best_roi_volume = backtest['roi']
                best_strategy_volume = {
                    'name': f'Best ROI with ≥{min_bets_per_year} bets/year',
                    'min_ev': ev,
                    'min_confidence': conf,
                    'bet_types': bet_types,
                    'expected_roi': backtest['roi'],
                    'expected_win_rate': backtest['win_rate'] * 100,
                    'expected_bets': backtest['total_bets'],
                    'expected_bets_per_year': bets_per_year,
                    'avg_ev': backtest['avg_ev'],
                    'avg_odds': backtest['avg_odds'],
                    'min_bets_per_year': min_bets_per_year
                }
    
    print(f"\nProgresssion : {total_combinations}/{total_combinations} (100.0%)")
    print(f"Stratégies testées : {len(results)}")
    print(f"  • Avec ROI > 0 : {len([r for r in results if r['roi'] > 0])}")
    print(f"  • Avec ≥{min_bets_per_year} paris/an : {len([r for r in results if r['meets_volume_requirement']])}")
    print(f"  • Avec ROI > 0 ET ≥{min_bets_per_year} paris/an : {len([r for r in results if r['roi'] > 0 and r['meets_volume_requirement']])}")
    
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    
    return best_strategy_pure, best_strategy_volume, results_df


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         STRATEGY OPTIMIZER - MAXIMISER LE ROI                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    with_xg = False
    
    # 1. Charger le modèle
    print(f"\nChargement du modèle...")
    model_path = SavePaths.get_latest_model('production', with_xg=with_xg)
    
    if model_path is None:
        print("   Pas de modèle calibré, recherche du modèle optimisé...")
        model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)
    
    if model_path is None:
        print(f"\nAucun modèle trouvé")
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
    
    # 3. Optimiser la stratégie
    best_strategy_pure, best_strategy_volume, results_df = optimize_strategy(
        test_df, model, features, 
        min_bets_per_year=100  # Minimum 100 paris par an
    )
    
    # 4. Afficher les résultats
    if best_strategy_pure:
        print(f"\n{'='*70}")
        print(f"  STRATÉGIE 1 : MEILLEUR ROI PUR ET DUR")
        print(f"{'='*70}")
        
        print(f"\nParamètres optimaux :")
        print(f"   • EV minimum       : {best_strategy_pure['min_ev']*100:.1f}%")
        print(f"   • Conf minimum     : {best_strategy_pure['min_confidence']*100:.1f}%")
        print(f"   • Types de paris   : {best_strategy_pure['bet_types'] or 'Tous'}")
        
        print(f"\nPerformances attendues :")
        print(f"   • ROI              : {best_strategy_pure['expected_roi']:+.2f}%")
        print(f"   • Win rate         : {best_strategy_pure['expected_win_rate']:.1f}%")
        print(f"   • Nombre de paris  : {best_strategy_pure['expected_bets']}")
        print(f"   • Paris par an     : {best_strategy_pure['expected_bets_per_year']:.0f}")
        print(f"   • EV moyen         : {best_strategy_pure['avg_ev']:+.2f}%")
        print(f"   • Cote moyenne     : {best_strategy_pure['avg_odds']:.2f}")
        
        # Profit total
        profit_total = (best_strategy_pure['expected_bets'] * 10 * best_strategy_pure['expected_roi']) / 100
        print(f"   • Profit total     : {profit_total:+.0f}€ (mise 10€/pari)")
        
        # Sauvegarder la stratégie ROI pur
        strategy_path = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy_pure.json',
            with_xg=with_xg
        )
        
        with open(strategy_path, 'w') as f:
            json.dump(best_strategy_pure, f, indent=4)
        
        print(f"\nStratégie sauvegardée : {strategy_path}")
    
    if best_strategy_volume:
        print(f"\n{'='*70}")
        print(f"  STRATÉGIE 2 : MEILLEUR ROI AVEC ≥100 PARIS/AN")
        print(f"{'='*70}")
        
        print(f"\nParamètres optimaux :")
        print(f"   • EV minimum       : {best_strategy_volume['min_ev']*100:.1f}%")
        print(f"   • Conf minimum     : {best_strategy_volume['min_confidence']*100:.1f}%")
        print(f"   • Types de paris   : {best_strategy_volume['bet_types'] or 'Tous'}")
        
        print(f"\nPerformances attendues :")
        print(f"   • ROI              : {best_strategy_volume['expected_roi']:+.2f}%")
        print(f"   • Win rate         : {best_strategy_volume['expected_win_rate']:.1f}%")
        print(f"   • Nombre de paris  : {best_strategy_volume['expected_bets']}")
        print(f"   • Paris par an     : {best_strategy_volume['expected_bets_per_year']:.0f}")
        print(f"   • Minimum requis   : {best_strategy_volume['min_bets_per_year']} paris/an")
        print(f"   • EV moyen         : {best_strategy_volume['avg_ev']:+.2f}%")
        print(f"   • Cote moyenne     : {best_strategy_volume['avg_odds']:.2f}")
        
        # Profit total
        profit_total = (best_strategy_volume['expected_bets'] * 10 * best_strategy_volume['expected_roi']) / 100
        print(f"   • Profit total     : {profit_total:+.0f}€ (mise 10€/pari)")
        
        # Sauvegarder la stratégie avec volume
        strategy_path_volume = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy_volume.json',
            with_xg=with_xg
        )
        
        with open(strategy_path_volume, 'w') as f:
            json.dump(best_strategy_volume, f, indent=4)
        
        print(f"\nStratégie sauvegardée : {strategy_path_volume}")
        
        # Utiliser la stratégie volume comme stratégie par défaut
        strategy_path_default = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy.json',
            with_xg=with_xg
        )
        
        with open(strategy_path_default, 'w') as f:
            json.dump(best_strategy_volume, f, indent=4)
        
        print(f"Stratégie par défaut : {strategy_path_default}")
    
    # 5. Comparaison
    if best_strategy_pure and best_strategy_volume:
        print(f"\n{'='*70}")
        print(f"  COMPARAISON DES STRATÉGIES")
        print(f"{'='*70}\n")
        
        print(f"{'Critère':<25} {'Pure (ROI Max)':<20} {'Volume (≥100/an)':<20}")
        print(f"{'-'*70}")
        print(f"{'ROI':<25} {best_strategy_pure['expected_roi']:>18.2f}%  {best_strategy_volume['expected_roi']:>18.2f}%")
        print(f"{'Win Rate':<25} {best_strategy_pure['expected_win_rate']:>18.1f}%  {best_strategy_volume['expected_win_rate']:>18.1f}%")
        print(f"{'Paris total':<25} {best_strategy_pure['expected_bets']:>18.0f}  {best_strategy_volume['expected_bets']:>18.0f}")
        print(f"{'Paris/an':<25} {best_strategy_pure['expected_bets_per_year']:>18.0f}  {best_strategy_volume['expected_bets_per_year']:>18.0f}")
        
        profit_pure = (best_strategy_pure['expected_bets'] * 10 * best_strategy_pure['expected_roi']) / 100
        profit_volume = (best_strategy_volume['expected_bets'] * 10 * best_strategy_volume['expected_roi']) / 100
        print(f"{'Profit total (10€/pari)':<25} {profit_pure:>17.0f}€  {profit_volume:>17.0f}€")
        
        if profit_pure > profit_volume:
            print(f"\n   Stratégie PURE recommandée (profit absolu supérieur)")
        else:
            print(f"\n   Stratégie VOLUME recommandée (profit absolu supérieur)")
        
        # 6. Sauvegarder tous les résultats
        if len(results_df) > 0:
            results_df = results_df.sort_values('roi', ascending=False)
            
            results_path = SavePaths.get_result_path(
                category='production/strategy',
                filename='strategy_optimization_results.csv',
                with_xg=with_xg
            )
            results_df.to_csv(results_path, index=False)
            
            print(f"\nTous les résultats sauvegardés : {results_path}")
            
            # Top 10
            print(f"\nTop 10 des meilleures stratégies (par ROI) :")
            print(results_df.head(10)[['min_ev', 'min_confidence', 'bet_types', 'total_bets', 'bets_per_year', 'roi']].to_string(index=False))
            
            # Top 10 avec volume
            results_volume = results_df[results_df['meets_volume_requirement'] == True].copy()
            if len(results_volume) > 0:
                print(f"\nTop 10 des meilleures stratégies (≥100 paris/an) :")
                print(results_volume.head(10)[['min_ev', 'min_confidence', 'bet_types', 'total_bets', 'bets_per_year', 'roi']].to_string(index=False))
        
    else:
        print(f"\nAucune stratégie profitable trouvée")
        print(f"Essaye de :")
        print(f"  • Améliorer ton modèle (step2b_optimization.py)")
        print(f"  • Calibrer ton modèle (step2c_calibration.py)")
        print(f"  • Récupérer de meilleures cotes (OddsPortal)")
    
    # 4. Afficher les résultats
    if best_strategy_pure:
        print(f"\n{'='*70}")
        print(f"  STRATÉGIE 1 : MEILLEUR ROI ABSOLU")
        print(f"{'='*70}")
        
        print(f"\nParamètres optimaux :")
        print(f"   • EV minimum       : {best_strategy_pure['min_ev']*100:.1f}%")
        print(f"   • Conf minimum     : {best_strategy_pure['min_confidence']*100:.1f}%")
        print(f"   • Types de paris   : {best_strategy_pure['bet_types'] or 'Tous'}")
        
        print(f"\nPerformances attendues :")
        print(f"   • ROI              : {best_strategy_pure['expected_roi']:+.2f}%")
        print(f"   • Win rate         : {best_strategy_pure['expected_win_rate']:.1f}%")
        print(f"   • Nombre de paris  : {best_strategy_pure['expected_bets']}")
        print(f"   • EV moyen         : {best_strategy_pure['avg_ev']:+.2f}%")
        print(f"   • Cote moyenne     : {best_strategy_pure['avg_odds']:.2f}")
        
        # Sauvegarder la stratégie ROI max
        strategy_path = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy_roi.json',
            with_xg=with_xg
        )
        
        with open(strategy_path, 'w') as f:
            json.dump(best_strategy_pure, f, indent=4)
        
        print(f"\nStratégie sauvegardée : {strategy_path}")
    
    if best_strategy_volume:
        print(f"\n{'='*70}")
        print(f"  STRATÉGIE 2 : MEILLEUR ROI AVEC VOLUME CIBLE (1/3)")
        print(f"{'='*70}")
        
        print(f"\nParamètres optimaux :")
        print(f"   • EV minimum       : {best_strategy_volume['min_ev']*100:.1f}%")
        print(f"   • Conf minimum     : {best_strategy_volume['min_confidence']*100:.1f}%")
        print(f"   • Types de paris   : {best_strategy_volume['bet_types'] or 'Tous'}")
        
        print(f"\nPerformances attendues :")
        print(f"   • ROI              : {best_strategy_volume['expected_roi']:+.2f}%")
        print(f"   • Win rate         : {best_strategy_volume['expected_win_rate']:.1f}%")
        print(f"   • Nombre de paris  : {best_strategy_volume['expected_bets']}")
        print(f"   • Objectif (1/3)   : {best_strategy_volume['target_bets']}")
        print(f"   • Écart            : {best_strategy_volume['distance_to_target']} paris")
        print(f"   • EV moyen         : {best_strategy_volume['avg_ev']:+.2f}%")
        print(f"   • Cote moyenne     : {best_strategy_volume['avg_odds']:.2f}")
        
        # Sauvegarder la stratégie avec volume
        strategy_path_volume = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy_volume.json',
            with_xg=with_xg
        )
        
        with open(strategy_path_volume, 'w') as f:
            json.dump(best_strategy_volume, f, indent=4)
        
        print(f"\nStratégie sauvegardée : {strategy_path_volume}")
        
        # Utiliser la stratégie volume comme stratégie par défaut
        strategy_path_default = SavePaths.get_result_path(
            category='production/strategy',
            filename='best_strategy.json',
            with_xg=with_xg
        )
        
        with open(strategy_path_default, 'w') as f:
            json.dump(best_strategy_volume, f, indent=4)
        
        print(f"Stratégie par défaut : {strategy_path_default}")
        
        # 5. Sauvegarder tous les résultats
        if len(results_df) > 0:
            results_df = results_df.sort_values('roi', ascending=False)
            
            results_path = SavePaths.get_result_path(
                category='production/strategy',
                filename='strategy_optimization_results.csv',
                with_xg=with_xg
            )
            results_df.to_csv(results_path, index=False)
            
            print(f"\nTous les résultats sauvegardés : {results_path}")
            
            # Top 10
            print(f"\nTop 10 des meilleures stratégies (par ROI) :")
            print(results_df.head(10)[['min_ev', 'min_confidence', 'bet_types', 'total_bets', 'win_rate', 'roi']].to_string(index=False))
            
            # Top 10 proches de l'objectif volume
            if 'distance_to_target' in results_df.columns:
                results_volume = results_df[results_df['roi'] > 0].sort_values('distance_to_target')
                if len(results_volume) > 0:
                    print(f"\nTop 10 des stratégies proches de l'objectif 1/3 (ROI > 0) :")
                    print(results_volume.head(10)[['min_ev', 'min_confidence', 'bet_types', 'total_bets', 'distance_to_target', 'roi']].to_string(index=False))
        
    else:
        print(f"\nAucune stratégie profitable trouvée")
        print(f"Essaye de :")
        print(f"  • Améliorer ton modèle (step2b_optimization.py)")
        print(f"  • Calibrer ton modèle (step2c_calibration.py)")
        print(f"  • Récupérer de meilleures cotes (OddsPortal)")
    
    print(f"\n{'='*70}")
    print(f"  OPTIMISATION TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()