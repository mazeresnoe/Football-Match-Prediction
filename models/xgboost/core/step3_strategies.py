"""
Step 3 Advanced : Test de multiples stratégies de paris
Compare différents niveaux de sélectivité pour trouver la stratégie optimale
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths

sns.set_style("whitegrid")


def calculate_expected_value(model_probs, bookmaker_odds):
    """Calcule l'Expected Value pour chaque issue"""
    ev_home = (model_probs[0] * bookmaker_odds[0]) - 1
    ev_draw = (model_probs[1] * bookmaker_odds[1]) - 1
    ev_away = (model_probs[2] * bookmaker_odds[2]) - 1
    return ev_home, ev_draw, ev_away


def find_value_bets_strategy(df, model, features, min_ev, min_confidence, bet_types=None):
    """
    Trouve les value bets selon une stratégie donnée
    
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
    """Backtest d'une stratégie"""
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


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║      STEP 3 : TEST DE STRATÉGIES MULTIPLES DE PARIS          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    # Configuration
    with_xg = False
    
    # 1. Charger le modèle calibré
    print(f"\n Chargement du modèle...")

    # Essayer de charger depuis production (modèle calibré)
    model_path = SavePaths.get_latest_model('production', with_xg=with_xg)

    # Si pas trouvé, essayer dans experiments (modèle optimisé)
    if model_path is None:
        print("   ⚠️ Pas de modèle calibré, recherche du modèle optimisé...")
        model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)

    if model_path is None:
        print(f"❌ Aucun modèle trouvé")
        print("   Exécute d'abord 'step2c_calibration.py' ou 'step2b_optimization.py'")
        return

    print(f"   Chargement depuis : {model_path}")
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    print(f"✅ Modèle chargé : {model_path.name}")
    
    # 2. Charger les données
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    
    # 3. Définir les stratégies à tester
    strategies = [
        # Stratégie 1 : Conservatrice (très sélective)
        {
            'name': 'Conservative',
            'min_ev': 0.10,
            'min_confidence': 0.55,
            'bet_types': ['Home', 'Away']  # Pas de draws
        },
        # Stratégie 2 : Modérée
        {
            'name': 'Moderate',
            'min_ev': 0.08,
            'min_confidence': 0.50,
            'bet_types': None  # Tous types
        },
        # Stratégie 3 : Home/Away seulement
        {
            'name': 'No Draws',
            'min_ev': 0.06,
            'min_confidence': 0.48,
            'bet_types': ['Home', 'Away']
        },
        # Stratégie 4 : High EV
        {
            'name': 'High EV',
            'min_ev': 0.12,
            'min_confidence': 0.45,
            'bet_types': None
        },
        # Stratégie 5 : High Confidence
        {
            'name': 'High Confidence',
            'min_ev': 0.05,
            'min_confidence': 0.60,
            'bet_types': None
        },
        # Stratégie 6 : Balanced
        {
            'name': 'Balanced',
            'min_ev': 0.07,
            'min_confidence': 0.52,
            'bet_types': ['Home', 'Away']
        },
    ]
    
    # 4. Tester chaque stratégie
    print(f"\n{'='*70}")
    print(f"  TEST DES STRATÉGIES SUR LE TEST SET")
    print(f"{'='*70}\n")
    
    results = []
    
    for strategy in strategies:
        print(f"   Test : {strategy['name']}")
        print(f"   • EV min : {strategy['min_ev']*100:.0f}%")
        print(f"   • Conf min : {strategy['min_confidence']*100:.0f}%")
        print(f"   • Types : {strategy['bet_types'] or 'Tous'}")
        
        # Trouver les value bets
        value_bets = find_value_bets_strategy(
            test_df, model, features,
            min_ev=strategy['min_ev'],
            min_confidence=strategy['min_confidence'],
            bet_types=strategy['bet_types']
        )
        
        if len(value_bets) == 0:
            print(f"   ❌ Aucun value bet trouvé\n")
            continue
        
        # Backtest
        backtest = backtest_strategy(value_bets, stake_per_bet=10)
        
        if backtest:
            print(f"      Résultats :")
            print(f"      • Paris : {backtest['total_bets']}")
            print(f"      • Win rate : {backtest['win_rate']*100:.1f}%")
            print(f"      • ROI : {backtest['roi']:+.2f}%")
            print(f"      • Profit : {backtest['total_profit']:+.0f}€")
            
            results.append({
                'strategy': strategy['name'],
                'min_ev': strategy['min_ev'] * 100,
                'min_confidence': strategy['min_confidence'] * 100,
                'bet_types': str(strategy['bet_types']) if strategy['bet_types'] else 'All',
                'total_bets': backtest['total_bets'],
                'win_rate': backtest['win_rate'] * 100,
                'roi': backtest['roi'],
                'profit': backtest['total_profit'],
                'avg_ev': backtest['avg_ev'],
                'avg_odds': backtest['avg_odds'],
            })
            
            if backtest['roi'] > 0:
                print(f"       ROI POSITIF !\n")
            else:
                print()
    
    # 5. Comparaison des stratégies
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('roi', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"  CLASSEMENT DES STRATÉGIES (par ROI)")
        print(f"{'='*70}\n")
        print(results_df[['strategy', 'total_bets', 'win_rate', 'roi', 'profit']].to_string(index=False))
        
        # Meilleure stratégie
        best = results_df.iloc[0]
        print(f"\n MEILLEURE STRATÉGIE : {best['strategy']}")
        print(f"   • ROI : {best['roi']:+.2f}%")
        print(f"   • Nombre de paris : {best['total_bets']:.0f}")
        print(f"   • Win rate : {best['win_rate']:.1f}%")
        print(f"   • Profit total : {best['profit']:+.0f}€")
        
        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. ROI par stratégie
        ax = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in results_df['roi']]
        ax.barh(results_df['strategy'], results_df['roi'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('ROI (%)')
        ax.set_title('ROI par stratégie', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Win rate vs ROI
        ax = axes[0, 1]
        scatter = ax.scatter(results_df['win_rate'], results_df['roi'], 
                           s=results_df['total_bets']*2, 
                           c=results_df['roi'], cmap='RdYlGn', alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel('ROI (%)')
        ax.set_title('Win Rate vs ROI', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='ROI')
        
        # Annoter les points
        for idx, row in results_df.iterrows():
            ax.annotate(row['strategy'], 
                       (row['win_rate'], row['roi']),
                       fontsize=8, alpha=0.7)
        
        # 3. Nombre de paris vs ROI
        ax = axes[1, 0]
        ax.scatter(results_df['total_bets'], results_df['roi'], 
                  s=100, c=results_df['roi'], cmap='RdYlGn', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Nombre de paris')
        ax.set_ylabel('ROI (%)')
        ax.set_title('Sélectivité vs ROI', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 4. Stats de la meilleure stratégie
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
MEILLEURE STRATÉGIE
{best['strategy']}

Paramètres :
• EV min : {best['min_ev']:.0f}%
• Conf min : {best['min_confidence']:.0f}%
• Types : {best['bet_types']}

Résultats :
• Total paris : {best['total_bets']:.0f}
• Win rate : {best['win_rate']:.1f}%
• ROI : {best['roi']:+.2f}%
• Profit : {best['profit']:+.0f}€
• Misé : {best['total_bets']*10:.0f}€
        """
        
        color = 'green' if best['roi'] > 0 else 'red'
        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor=color, 
                                           alpha=0.1, edgecolor=color, linewidth=2))
        
        plt.suptitle('ANALYSE DES STRATÉGIES DE PARIS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarder en production
        plot_path = SavePaths.get_result_path(
            category='production',
            filename='strategies_comparison.png',
            with_xg=with_xg
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        # Copie historique
        history_plot_path = SavePaths.get_result_path(
            category='step3_strategies',
            filename='strategies_comparison.png',
            with_xg=with_xg,
            use_date_folder=True
        )
        plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')

        print(f"\n✅ Graphique sauvegardé : {plot_path}")
        print(f"   Historique : {history_plot_path}")
        plt.close()
        
        # Sauvegarder les résultats EN PRODUCTION
        result_path = SavePaths.get_result_path(
            category='production',
            filename='strategies_results.csv',
            with_xg=with_xg,
            use_date_folder=False  # Écraser le fichier à chaque fois
        )
        results_df.to_csv(result_path, index=False)

        # Aussi garder une copie historique
        history_path = SavePaths.get_result_path(
            category='step3_strategies',
            filename='strategies_results.csv',
            with_xg=with_xg,
            use_date_folder=True  # Créer un dossier par date
        )
        results_df.to_csv(history_path, index=False)

        # Sauvegarder les métadonnées
        best_strategy = results_df.iloc[0]
        SavePaths.save_metadata(
            category='production',
            filename='strategies_results.csv',
            metadata={
                'best_strategy': best_strategy['strategy'],
                'best_roi': float(best_strategy['roi']),
                'total_strategies_tested': len(results_df),
                'positive_roi_count': int((results_df['roi'] > 0).sum())
            },
            with_xg=with_xg
        )

        print(f"✅ Résultats sauvegardés :")
        print(f"   Production : {result_path}")
        print(f"   Historique : {history_path}")
    
    else:
        print(f"\n❌ Aucune stratégie n'a trouvé de value bets")
        print(f"   → Les critères sont peut-être trop stricts")
        print(f"   → Essaye de réduire min_ev ou min_confidence")
    
    print(f"\n{'='*70}")
    print(f"  ANALYSE TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()