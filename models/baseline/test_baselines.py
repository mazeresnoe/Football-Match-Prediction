"""
Script de test pour les baselines sur les VRAIES données

Ce script :
1. Charge tes données réelles (AVEC et SANS XG)
2. Affiche des statistiques détaillées
3. Fait le split Train/CV/Test
4. Évalue les 3 baselines
5. Compare les performances

Usage:
    python test_baselines.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import des modules locaux
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as configs
import models.utils as utils
from baseline_bookmaker import BookmakerProportionalBaseline, BookmakerSimpleBaseline
from baseline_simple import SimpleEloBaseline
from baseline_logreg import LogisticRegressionBaseline


def analyze_dataset(df: pd.DataFrame, dataset_name: str = ""):
    """
    Affiche des statistiques détaillées sur le dataset
    """
    print(f"\n{'='*70}")
    print(f"  ANALYSE DU DATASET {dataset_name}")
    print(f"{'='*70}\n")
    
    # 1. Dimensions
    print(f" Dimensions :")
    print(f"   • Nombre de matchs : {len(df):,}")
    print(f"   • Nombre de colonnes : {len(df.columns)}")
    
    # 2. Période couverte
    print(f"\n Période :")
    print(f"   • Date min : {df['date'].min()}")
    print(f"   • Date max : {df['date'].max()}")
    print(f"   • Durée : {(df['date'].max() - df['date'].min()).days} jours")
    
    # 3. Distribution des résultats
    print(f"\n Distribution des résultats :")
    result_counts = df['result'].value_counts()
    for result, count in result_counts.items():
        result_name = {1: "Home Win", 0: "Draw", -1: "Away Win"}[result]
        pct = count / len(df) * 100
        print(f"   • {result_name:10} : {count:6,} matchs ({pct:5.1f}%)")
    
    # 4. Ligues
    if 'league' in df.columns:
        print(f"\n Ligues (Top 10) :")
        league_counts = df['league'].value_counts().head(10)
        for league, count in league_counts.items():
            print(f"   • {league:25} : {count:5,} matchs")
    
    # 5. Odds disponibles
    if 'odds_home' in df.columns:
        n_with_odds = df['odds_home'].notna().sum()
        print(f"\n Odds :")
        print(f"   • Matchs avec odds : {n_with_odds:,} ({n_with_odds/len(df)*100:.1f}%)")
        if n_with_odds > 0:
            print(f"   • Odds Home moyenne : {df['odds_home'].mean():.2f}")
            print(f"   • Odds Draw moyenne : {df['odds_draw'].mean():.2f}")
            print(f"   • Odds Away moyenne : {df['odds_away'].mean():.2f}")
    
    # 6. Features importantes disponibles
    print(f"\n Features clés :")
    important_features = [
        'elo_home', 'elo_away', 'elo_diff',
        'home_form_5', 'away_form_5',
        'home_goals_scored_avg_5', 'away_goals_scored_avg_5'
    ]
    for feat in important_features:
        if feat in df.columns:
            print(f" {feat:30} : Moyenne = {df[feat].mean():.2f}")
        else:
            print(f" {feat:30} : MANQUANT")


def show_sample_predictions(model, model_name: str, df: pd.DataFrame, n: int = 5):
    """
    Affiche quelques prédictions pour visualiser le comportement du modèle
    """
    print(f"\n{'='*70}")
    print(f"  EXEMPLES DE PRÉDICTIONS - {model_name}")
    print(f"{'='*70}\n")
    
    # Prendre n matchs aléatoires
    sample = df.sample(n=min(n, len(df)), random_state=42)
    probs = model.predict_proba(sample)
    
    print(f"{'Home':15} vs {'Away':15} | {'Prob H':>8} {'Prob D':>8} {'Prob A':>8} | {'Pred':>6} | {'Réel':>6} | OK?")
    print(f"{'-'*70}")
    
    for i, (idx, row) in enumerate(sample.iterrows()):
        home = row['home_team'][:15] if 'home_team' in row else f"Team {i}H"
        away = row['away_team'][:15] if 'away_team' in row else f"Team {i}A"
        
        pred_class = np.argmax(probs[i])
        pred_name = ['Home', 'Draw', 'Away'][pred_class]
        
        actual = {1: 'Home', 0: 'Draw', -1: 'Away'}[row['result']]
        correct = "✓" if pred_name == actual else "✗"
        
        print(f"{home:15} vs {away:15} | {probs[i][0]:8.3f} {probs[i][1]:8.3f} {probs[i][2]:8.3f} "
              f"| {pred_name:>6} | {actual:>6} | {correct}")


def evaluate_baselines(with_xg: bool = False):
    """
    Évalue les 3 baselines sur un dataset (avec ou sans XG)
    
    Args:
        with_xg: Si True, utilise le dataset avec XG
    
    Returns:
        DataFrame avec les résultats
    """
    dataset_name = "AVEC XG" if with_xg else "SANS XG"
    
    print(f"\n{'#'*70}")
    print(f"#{'  ÉVALUATION ' + dataset_name:^68}#")
    print(f"{'#'*70}")
    
    # ========================================
    # 1. CHARGER LES DONNÉES
    # ========================================
    print(f"\n🔄 Chargement des données...")
    try:
        df = utils.load_data(with_xg=with_xg, merge_odds=True)
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR : {e}")
        print(f"\n Vérifie que le fichier existe")
        return None
    
    # ========================================
    # 2. ANALYSER LE DATASET
    # ========================================
    analyze_dataset(df, dataset_name)
    
    # ========================================
    # 3. SPLIT TRAIN/CV/TEST
    # ========================================
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    
    # Ground truth
    y_test = test_df[configs.TARGET_COL].values
    
    # ========================================
    # 4. ÉVALUER LES 3 BASELINES
    # ========================================
    
    all_results = []
    
    # -----------------------------------------
    # BASELINE 1A : BOOKMAKER SIMPLE
    # -----------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  BASELINE 1A : BOOKMAKER (SIMPLE)")
    print(f"{'='*70}")

    bookmaker_simple = BookmakerSimpleBaseline()

    # Filtrer pour ne garder que les matchs avec odds
    test_with_odds = test_df[test_df['odds_home'].notna()].copy()
    y_test_with_odds = test_with_odds[configs.TARGET_COL].values
    odds_test_with_odds = test_with_odds[['odds_home', 'odds_draw', 'odds_away']]

    print(f"\n⚠️  Évalué sur {len(test_with_odds):,} matchs avec odds")

    # Test
    probs_test = bookmaker_simple.predict_proba(test_with_odds)
    res_test = utils.evaluate_predictions(y_test_with_odds, probs_test, odds_test_with_odds, "Bookmaker (Simple)")
    res_test['dataset'] = 'test'
    res_test['n_matches_evaluated'] = len(test_with_odds)
    res_test['with_xg'] = with_xg
    utils.print_evaluation_summary(res_test, "TEST")
    all_results.append(res_test)

    # -----------------------------------------
    # BASELINE 1B : BOOKMAKER PROPORTIONNEL
    # -----------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  BASELINE 1B : BOOKMAKER (PROPORTIONNEL)")
    print(f"{'='*70}")

    bookmaker_prop = BookmakerProportionalBaseline()

    print(f"\n⚠️  Évalué sur {len(test_with_odds):,} matchs avec odds")

    # Test
    probs_test = bookmaker_prop.predict_proba(test_with_odds)
    res_test = utils.evaluate_predictions(y_test_with_odds, probs_test, odds_test_with_odds, "Bookmaker (Proportional)")
    res_test['dataset'] = 'test'
    res_test['n_matches_evaluated'] = len(test_with_odds)
    res_test['with_xg'] = with_xg
    utils.print_evaluation_summary(res_test, "TEST")
    all_results.append(res_test)

    # -----------------------------------------
    # BASELINE 2 : SIMPLE ELO
    # -----------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  BASELINE 2 : SIMPLE ELO")
    print(f"{'='*70}")
    
    simple_elo = SimpleEloBaseline(home_advantage=100)
    
    print(f"\n Évalué sur TOUS les {len(test_df):,} matchs\n")
    
    # Exemples de prédictions
    show_sample_predictions(simple_elo, "Simple Elo", test_df, n=5)
    
    # Test
    probs_test = simple_elo.predict_proba(test_df)
    res_test = utils.evaluate_predictions(y_test, probs_test, test_df[['odds_home', 'odds_draw', 'odds_away']], "Simple Elo")
    res_test['dataset'] = 'test'
    res_test['n_matches_evaluated'] = len(test_df)
    res_test['with_xg'] = with_xg
    utils.print_evaluation_summary(res_test, "TEST")
    all_results.append(res_test)
    
    # -----------------------------------------
    # BASELINE 3 : LOGISTIC REGRESSION
    # -----------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  BASELINE 3 : LOGISTIC REGRESSION")
    print(f"{'='*70}")
    
    # Choisir les features selon le dataset
    features = configs.LOGREG_FEATURES_WITH_XG if with_xg else configs.LOGREG_FEATURES_NO_XG
    logreg = LogisticRegressionBaseline(features=features)
    
    # Entraîner
    logreg.fit(train_df)
    
    print(f"\n Évalué sur TOUS les {len(test_df):,} matchs\n")
    
    # Exemples de prédictions
    show_sample_predictions(logreg, "Logistic Regression", test_df, n=5)
    
    # Test
    probs_test = logreg.predict_proba(test_df)
    res_test = utils.evaluate_predictions(y_test, probs_test, test_df[['odds_home', 'odds_draw', 'odds_away']], "Logistic Regression")
    res_test['dataset'] = 'test'
    res_test['n_matches_evaluated'] = len(test_df)
    res_test['with_xg'] = with_xg
    utils.print_evaluation_summary(res_test, "TEST")
    all_results.append(res_test)
    
    # ========================================
    # 5. COMPARAISON LOCALE
    # ========================================
    print(f"\n\n{'='*70}")
    print(f"  COMPARAISON - {dataset_name}")
    print(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    print(f"{'Modèle':<30} {'N matchs':>10} {'Accuracy':>10} {'Log Loss':>10} {'ROI (%)':>10}")
    print(f"{'-'*75}")
    for _, row in comparison_df.iterrows():
        n_eval = row.get('n_matches_evaluated', len(test_df))
        print(f"{row['model']:<30} {n_eval:>10,} {row['accuracy']:>10.4f} {row['log_loss']:>10.4f} {row['roi']:>10.2f}")
    
    # Sauvegarder
    filename = f"step1_baselines/baseline_comparison_{'with_xg' if with_xg else 'no_xg'}.csv"
    utils.save_results(all_results, filename, with_xg=with_xg)
    
    return comparison_df


def main():
    """
    Script principal de test
    """
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       TEST DES BASELINES - AVEC ET SANS XG                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================
    # ÉVALUER SANS XG
    # ========================================
    print(f"\n")
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║                     DATASET SANS XG                          ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    results_no_xg = evaluate_baselines(with_xg=False)
    
    # ========================================
    # ÉVALUER AVEC XG
    # ========================================
    print(f"\n\n")
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║                     DATASET AVEC XG                          ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    results_with_xg = evaluate_baselines(with_xg=True)
    
    # ========================================
    # COMPARAISON GLOBALE
    # ========================================
    if results_no_xg is not None and results_with_xg is not None:
        print(f"\n\n{'#'*70}")
        print(f"#{'  COMPARAISON GLOBALE : AVEC vs SANS XG':^68}#")
        print(f"{'#'*70}\n")
        
        # Combiner les résultats
        results_no_xg['dataset_type'] = 'NO_XG'
        results_with_xg['dataset_type'] = 'WITH_XG'
        
        all_results = pd.concat([results_no_xg, results_with_xg], ignore_index=True)
        
        # Tableau comparatif
        print(f"{'Modèle':<35} {'Dataset':>10} {'N matchs':>10} {'Accuracy':>10} {'Log Loss':>10} {'ROI (%)':>10}")
        print(f"{'-'*100}")
        
        # Trouver tous les modèles uniques
        unique_models = all_results['model'].unique()
        
        for model_name in sorted(unique_models):
            # Résultats NO_XG
            res_no_xg_filter = all_results[(all_results['model'] == model_name) & (all_results['dataset_type'] == 'NO_XG')]
            
            if len(res_no_xg_filter) > 0:
                res_no_xg = res_no_xg_filter.iloc[0]
                print(f"{model_name:<35} {'NO_XG':>10} {res_no_xg['n_matches_evaluated']:>10,} "
                    f"{res_no_xg['accuracy']:>10.4f} {res_no_xg['log_loss']:>10.4f} {res_no_xg['roi']:>10.2f}")
            
            # Résultats WITH_XG
            res_with_xg_filter = all_results[(all_results['model'] == model_name) & (all_results['dataset_type'] == 'WITH_XG')]
            
            if len(res_with_xg_filter) > 0:
                res_with_xg = res_with_xg_filter.iloc[0]
                print(f"{model_name:<35} {'WITH_XG':>10} {res_with_xg['n_matches_evaluated']:>10,} "
                    f"{res_with_xg['accuracy']:>10.4f} {res_with_xg['log_loss']:>10.4f} {res_with_xg['roi']:>10.2f}")
                
                # Différence (seulement si les deux existent)
                if len(res_no_xg_filter) > 0:
                    diff_acc = res_with_xg['accuracy'] - res_no_xg['accuracy']
                    diff_ll = res_with_xg['log_loss'] - res_no_xg['log_loss']
                    diff_roi = res_with_xg['roi'] - res_no_xg['roi']
                    
                    print(f"{'→ Différence':<35} {' ':>10} {' ':>10} "
                        f"{diff_acc:>+10.4f} {diff_ll:>+10.4f} {diff_roi:>+10.2f}")
                
                print(f"{'-'*100}")
        
        # Analyse
        print(f"\n📊 Analyse :")
        
        # LogReg avec XG vs sans XG
        logreg_no_xg_filter = all_results[(all_results['model'] == 'Logistic Regression') & (all_results['dataset_type'] == 'NO_XG')]
        logreg_with_xg_filter = all_results[(all_results['model'] == 'Logistic Regression') & (all_results['dataset_type'] == 'WITH_XG')]
        
        if len(logreg_no_xg_filter) > 0 and len(logreg_with_xg_filter) > 0:
            logreg_no_xg = logreg_no_xg_filter.iloc[0]
            logreg_with_xg = logreg_with_xg_filter.iloc[0]
            
            acc_gain = (logreg_with_xg['accuracy'] - logreg_no_xg['accuracy']) * 100
            ll_gain = logreg_no_xg['log_loss'] - logreg_with_xg['log_loss']
            roi_gain = logreg_with_xg['roi'] - logreg_no_xg['roi']
            
            print(f"\n   📈 Impact des Expected Goals (XG) sur LogReg :")
            print(f"      • Accuracy  : {acc_gain:+.2f} points de pourcentage")
            print(f"      • Log Loss  : {ll_gain:+.4f} ({'amélioration' if ll_gain > 0 else 'dégradation'})")
            print(f"      • ROI       : {roi_gain:+.2f}%")
            
            if acc_gain > 0.5:
                print(f"\n   ✅ Les XG apportent une valeur significative (+{acc_gain:.2f}% accuracy)")
            elif acc_gain > 0:
                print(f"\n   ⚙️  Les XG apportent une légère amélioration (+{acc_gain:.2f}% accuracy)")
            else:
                print(f"\n   ⚠️  Les XG n'apportent pas d'amélioration ({acc_gain:.2f}% accuracy)")
                print(f"       → Possibles raisons : overfitting, corrélation avec autres features, bruit")
        
        # Comparer les 2 méthodes Bookmaker
        print(f"\n   🎯 Comparaison Bookmaker Simple vs Proportionnel (NO_XG) :")
        
        bookmaker_simple_no_xg = all_results[(all_results['model'] == 'Bookmaker (Simple)') & (all_results['dataset_type'] == 'NO_XG')]
        bookmaker_prop_no_xg = all_results[(all_results['model'] == 'Bookmaker (Proportional)') & (all_results['dataset_type'] == 'NO_XG')]
        
        if len(bookmaker_simple_no_xg) > 0 and len(bookmaker_prop_no_xg) > 0:
            simple = bookmaker_simple_no_xg.iloc[0]
            prop = bookmaker_prop_no_xg.iloc[0]
            
            ll_diff = simple['log_loss'] - prop['log_loss']
            roi_diff = prop['roi'] - simple['roi']
            
            print(f"      • Log Loss  : {ll_diff:+.4f} (Proportionnel {'meilleur' if ll_diff > 0 else 'moins bon'})")
            print(f"      • ROI       : {roi_diff:+.2f}% (Proportionnel {'meilleur' if roi_diff > 0 else 'moins bon'})")
            
            if ll_diff > 0:
                print(f"\n   ✅ La méthode proportionnelle est effectivement meilleure (comme attendu)")
            else:
                print(f"\n   ⚠️  La méthode simple performe mieux (inattendu, vérifier les données)")


if __name__ == "__main__":
    sys.exit(main())