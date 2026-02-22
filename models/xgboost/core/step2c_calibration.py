"""
Step 2c : Calibration des probabilités
NOTE: Ce script n'est utile QUE pour les modèles SINGLE.
Les ensembles créés par step2b sont DÉJÀ calibrés.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths
from models.xgboost.utils.calibration import ManualCalibratedClassifier, ManualCalibratedEnsemble
from models.xgboost.core.step2b_optimization import XGBoostImproved

sns.set_style("whitegrid")


def plot_calibration_curve(y_true, probs_before, probs_after, with_xg: bool):
    """Compare les probabilités avant et après calibration"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convertir y_true en indices si nécessaire
    if not isinstance(y_true[0], (int, np.integer)):
        y_true_idx = np.array([{1: 0, 0: 1, -1: 2}[y] for y in y_true])
    else:
        y_true_idx = y_true
    
    outcomes = ['Home Win', 'Draw', 'Away Win']
    
    for i, (outcome, ax) in enumerate(zip(outcomes, axes)):
        # Données pour cette classe
        y_binary = (y_true_idx == i).astype(int)
        prob_before = probs_before[:, i]
        prob_after = probs_after[:, i]
        
        # Créer des bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculer les fréquences observées avant calibration
        freq_before = []
        for j in range(len(bins) - 1):
            mask = (prob_before >= bins[j]) & (prob_before < bins[j+1])
            if mask.sum() > 0:
                freq_before.append(y_binary[mask].mean())
            else:
                freq_before.append(np.nan)
        
        # Calculer les fréquences observées après calibration
        freq_after = []
        for j in range(len(bins) - 1):
            mask = (prob_after >= bins[j]) & (prob_after < bins[j+1])
            if mask.sum() > 0:
                freq_after.append(y_binary[mask].mean())
            else:
                freq_after.append(np.nan)
        
        # Tracer
        ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré', linewidth=2)
        ax.plot(bin_centers, freq_before, 'o-', label='Avant calibration', 
               color='red', linewidth=2, markersize=8)
        ax.plot(bin_centers, freq_after, 's-', label='Après calibration', 
               color='green', linewidth=2, markersize=8)
        
        ax.set_xlabel('Probabilité prédite', fontsize=11)
        ax.set_ylabel('Fréquence observée', fontsize=11)
        ax.set_title(f'Calibration : {outcome}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = SavePaths.get_result_path(
        category='step2c_calibration',
        filename='calibration_curves.png',
        with_xg=with_xg
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Courbes de calibration : {plot_path}")
    plt.close()


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║      STEP 2C : CALIBRATION DES PROBABILITÉS                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    with_xg = False
    
    # 1. Charger le modèle
    print("\n📂 Chargement du modèle...")
    
    # Chercher le modèle le plus récent dans experiments
    model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)
    
    # Fallback sur production si pas trouvé
    if model_path is None:
        print("   ⚠️ Pas de modèle dans experiments/, recherche dans production...")
        model_path = SavePaths.get_latest_model('production', with_xg=with_xg)
    
    if model_path is None:
        print(f"❌ Aucun modèle trouvé")
        print("   Exécute d'abord 'step2b_optimization.py'")
        return
    
    print(f"   Chargement depuis : {model_path}")
    model_data = joblib.load(model_path)
    
    # ========================================
    # VÉRIFIER SI C'EST UN ENSEMBLE OU SINGLE
    # ========================================
    
    if 'ensemble' in model_data:
        # C'est un ensemble déjà calibré
        print(f"\n{'='*70}")
        print(f"  ⚠️  MODÈLE ENSEMBLE DÉTECTÉ")
        print(f"{'='*70}")
        print(f"\n❌ Ce script n'est pas nécessaire pour les ensembles !")
        print(f"\nRaison :")
        print(f"  • Les ensembles créés par step2b_optimization.py")
        print(f"  • sont DÉJÀ calibrés avec Isotonic Regression")
        print(f"  • sur les données Train+CV combinées")
        print(f"\nAction :")
        print(f"  ✅ Utilisez directement l'ensemble sauvegardé")
        print(f"  ✅ Passez à optimize_strategy.py pour value bets")
        print(f"\nCe script est uniquement utile pour :")
        print(f"  • Les modèles SINGLE non calibrés")
        print(f"  • Les anciens modèles sans ensemble")
        print(f"\n{'='*70}\n")
        return
    
    elif 'model' in model_data:
        # C'est un single model
        xgb_model = model_data['model']
        features = model_data['features']
        print(f"✅ Modèle SINGLE chargé : {len(features)} features")
        print(f"\n⚠️  NOTE : Les ensembles (step2b) sont recommandés")
        print(f"   Continuons avec la calibration du modèle single...\n")
        
    else:
        print("❌ Format de modèle inconnu")
        return
    
    # 2. Charger les données
    print("📂 Chargement des données...")
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    print("✅ Données chargées\n")
    
    # 3. Préparer les données
    X_cv = cv_df[features].values
    y_cv = cv_df[cfg.TARGET_COL].values
    y_cv_idx = np.array([{1: 0, 0: 1, -1: 2}[y] for y in y_cv])
    
    X_test = test_df[features].values
    y_test = test_df[cfg.TARGET_COL].values
    y_test_idx = np.array([{1: 0, 0: 1, -1: 2}[y] for y in y_test])
    
    # 4. Prédictions AVANT calibration
    print("📊 Évaluation AVANT calibration...")
    probs_cv_before = xgb_model.predict_proba(X_cv)
    probs_test_before = xgb_model.predict_proba(X_test)
    
    # Métriques avant (CV)
    logloss_cv_before = log_loss(y_cv_idx, probs_cv_before)
    
    y_cv_onehot = np.zeros((len(y_cv_idx), 3))
    for i, y in enumerate(y_cv_idx):
        y_cv_onehot[i, y] = 1
    brier_cv_before = np.mean(np.sum((probs_cv_before - y_cv_onehot) ** 2, axis=1))
    
    print(f"   • Log Loss (CV)  : {logloss_cv_before:.4f}")
    print(f"   • Brier Score (CV): {brier_cv_before:.4f}")
    
    # 5. CALIBRATION MANUELLE
    print(f"\n🔧 Calibration isotonique...")
    
    calibrated_model = ManualCalibratedClassifier(xgb_model, method='isotonic')
    calibrated_model.fit(X_cv, y_cv_idx)
    
    print(f"✅ Calibration terminée (3 calibrateurs isotoniques créés)")
    
    # 6. Prédictions APRÈS calibration
    print("\n📊 Évaluation APRÈS calibration...")
    probs_cv_after = calibrated_model.predict_proba(X_cv)
    probs_test_after = calibrated_model.predict_proba(X_test)
    
    # Métriques après (CV)
    logloss_cv_after = log_loss(y_cv_idx, probs_cv_after)
    
    y_cv_onehot_after = np.zeros((len(y_cv_idx), 3))
    for i, y in enumerate(y_cv_idx):
        y_cv_onehot_after[i, y] = 1
    brier_cv_after = np.mean(np.sum((probs_cv_after - y_cv_onehot_after) ** 2, axis=1))
    
    print(f"   • Log Loss (CV)  : {logloss_cv_after:.4f}")
    print(f"   • Brier Score (CV): {brier_cv_after:.4f}")
    
    # 7. Comparaison
    print(f"\n{'='*70}")
    print(f"  AMÉLIORATION DE LA CALIBRATION (sur CV)")
    print(f"{'='*70}")
    
    logloss_improv = (logloss_cv_after - logloss_cv_before) / logloss_cv_before * 100
    brier_improv = (brier_cv_after - brier_cv_before) / brier_cv_before * 100
    
    print(f"Log Loss  : {logloss_cv_before:.4f} → {logloss_cv_after:.4f} " + 
          f"({logloss_improv:+.2f}%)")
    print(f"Brier     : {brier_cv_before:.4f} → {brier_cv_after:.4f} " + 
          f"({brier_improv:+.2f}%)")
    
    # 8. Évaluation sur TEST avec odds
    print(f"\n{'='*70}")
    print(f"  ÉVALUATION SUR TEST SET")
    print(f"{'='*70}")
    
    odds_test = test_df[['odds_home', 'odds_draw', 'odds_away']].copy()
    mask_odds = odds_test.notna().all(axis=1)
    
    if mask_odds.sum() > 0:
        # Avant calibration
        res_before = utils.evaluate_predictions(
            y_test[mask_odds],
            probs_test_before[mask_odds],
            odds_test[mask_odds],
            "XGBoost AVANT calibration"
        )
        
        # Après calibration
        res_after = utils.evaluate_predictions(
            y_test[mask_odds],
            probs_test_after[mask_odds],
            odds_test[mask_odds],
            "XGBoost APRÈS calibration"
        )
        
        print(f"\n📊 AVANT CALIBRATION")
        utils.print_evaluation_summary(res_before, "TEST")
        
        print(f"\n📊 APRÈS CALIBRATION")
        utils.print_evaluation_summary(res_after, "TEST")
        
        # Amélioration du ROI
        roi_improvement = res_after['roi'] - res_before['roi']
        print(f"\n📈 Amélioration du ROI : {roi_improvement:+.2f} points")
        
        if res_after['roi'] > 0:
            print(f"✅ EXCELLENT ! ROI POSITIF : {res_after['roi']:.2f}%")
        else:
            print(f"⚠️  ROI encore négatif : {res_after['roi']:.2f}%")
    
    # 9. Métriques TEST (sans odds)
    logloss_test_before = log_loss(y_test_idx, probs_test_before)
    logloss_test_after = log_loss(y_test_idx, probs_test_after)
    
    y_test_onehot = np.zeros((len(y_test_idx), 3))
    for i, y in enumerate(y_test_idx):
        y_test_onehot[i, y] = 1
    brier_test_before = np.mean(np.sum((probs_test_before - y_test_onehot) ** 2, axis=1))
    brier_test_after = np.mean(np.sum((probs_test_after - y_test_onehot) ** 2, axis=1))
    
    print(f"\n{'='*70}")
    print(f"  MÉTRIQUES SUR TEST SET")
    print(f"{'='*70}")
    print(f"Log Loss  : {logloss_test_before:.4f} → {logloss_test_after:.4f}")
    print(f"Brier     : {brier_test_before:.4f} → {brier_test_after:.4f}")
    
    # 10. Visualisation
    plot_calibration_curve(y_cv, probs_cv_before, probs_cv_after, with_xg)
    
    # 11. Sauvegarder le modèle calibré EN PRODUCTION
    SavePaths.archive_current_model('xgboost_calibrated', with_xg=with_xg)

    model_path = SavePaths.get_model_path(
        category='production',
        model_name='xgboost_calibrated',
        with_xg=with_xg
    )
    joblib.dump({
        'model': calibrated_model,
        'features': features,
        'method': 'isotonic_manual'
    }, model_path)

    print(f"\n✅ Modèle calibré sauvegardé : {model_path}")
    
    # 12. Sauvegarder les résultats
    results = pd.DataFrame([
        {
            'model': 'XGBoost Single',
            'dataset': 'test',
            'log_loss': logloss_test_before,
            'brier_score': brier_test_before,
            'roi': res_before['roi'] if mask_odds.sum() > 0 else None
        },
        {
            'model': 'XGBoost Calibré',
            'dataset': 'test',
            'log_loss': logloss_test_after,
            'brier_score': brier_test_after,
            'roi': res_after['roi'] if mask_odds.sum() > 0 else None
        }
    ])

    result_path = SavePaths.get_result_path(
        category='step2c_calibration',
        filename='calibration_results.csv',
        with_xg=with_xg
    )
    results.to_csv(result_path, index=False)

    if mask_odds.sum() > 0:
        SavePaths.save_metadata(
            category='step2c_calibration',
            filename='calibration_results.csv',
            metadata={
                'calibration_method': 'isotonic_manual',
                'roi_improvement': res_after['roi'] - res_before['roi'],
                'log_loss_improvement': logloss_test_after - logloss_test_before,
                'brier_improvement': brier_test_after - brier_test_before
            },
            with_xg=with_xg
        )

    print(f"✅ Résultats sauvegardés : {result_path}")
    
    print(f"\n{'='*70}")
    print(f"  CALIBRATION TERMINÉE ✅")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

