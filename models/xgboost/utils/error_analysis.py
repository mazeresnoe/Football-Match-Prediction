"""
Analyse d'erreur du modèle XGBoost
Identifie où le modèle performe bien/mal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths

sns.set_style("whitegrid")


def load_best_model(with_xg: bool = False):
    """Charge le meilleur modèle sauvegardé"""
    
    # Essayer de charger depuis production (calibré)
    model_path = SavePaths.get_latest_model('production', with_xg=with_xg)
    
    # Si pas trouvé, essayer experiments (optimisé)
    if model_path is None:
        print("   ⚠️ Pas de modèle calibré, recherche du modèle optimisé...")
        model_path = SavePaths.get_latest_model('experiments', with_xg=with_xg)
    
    if model_path is None:
        print(f"❌ Aucun modèle trouvé")
        print("   Exécute d'abord 'step2b_optimization.py'")
        return None
    
    model_data = joblib.load(model_path)
    print(f"✅ Modèle chargé : {model_path}")
    print(f"   • {len(model_data['features'])} features")
    print(f"   • Best iteration : {model_data.get('best_iteration', 'N/A')}")
    
    return model_data


def analyze_prediction_errors(df: pd.DataFrame, y_true: np.ndarray, 
                              probs: np.ndarray, dataset_name: str = "Test"):
    """Analyse les erreurs de prédiction"""
    
    # Prédictions
    y_pred_idx = np.argmax(probs, axis=1)
    y_pred = np.array([{0: 1, 1: 0, 2: -1}[p] for p in y_pred_idx])
    
    # Ajouter les prédictions au dataframe
    df = df.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['correct'] = (y_true == y_pred).astype(int)
    df['prob_home'] = probs[:, 0]
    df['prob_draw'] = probs[:, 1]
    df['prob_away'] = probs[:, 2]
    df['max_prob'] = np.max(probs, axis=1)
    
    print(f"\n{'='*70}")
    print(f"  ANALYSE DES ERREURS - {dataset_name}")
    print(f"{'='*70}\n")
    
    # 1. Taux de réussite global
    accuracy = df['correct'].mean()
    print(f"📊 Accuracy globale : {accuracy:.2%}")
    print(f"   • Prédictions correctes : {df['correct'].sum():,} / {len(df):,}")
    
    # 2. Taux de réussite par type de résultat
    print(f"\n📈 Taux de réussite par résultat réel :")
    for outcome, label in [(1, "Home Win"), (0, "Draw"), (-1, "Away Win")]:
        subset = df[df['y_true'] == outcome]
        acc = subset['correct'].mean() if len(subset) > 0 else 0
        print(f"   • {label:10s} : {acc:.2%} ({subset['correct'].sum()}/{len(subset)})")
    
    # 3. Matrice de confusion
    print(f"\n🔢 Matrice de confusion :")
    confusion = pd.crosstab(
        df['y_true'].map({1: 'Home', 0: 'Draw', -1: 'Away'}),
        df['y_pred'].map({1: 'Home', 0: 'Draw', -1: 'Away'}),
        rownames=['Réel'],
        colnames=['Prédit']
    )
    print(confusion)
    
    # 4. Confiance du modèle selon la justesse
    print(f"\n🎯 Confiance moyenne du modèle :")
    conf_correct = df[df['correct'] == 1]['max_prob'].mean()
    conf_incorrect = df[df['correct'] == 0]['max_prob'].mean()
    print(f"   • Sur prédictions correctes   : {conf_correct:.3f}")
    print(f"   • Sur prédictions incorrectes : {conf_incorrect:.3f}")
    print(f"   • Différence (calibration)    : {conf_correct - conf_incorrect:.3f}")
    
    # 5. Analyse par ligue
    if 'league' in df.columns:
        print(f"\n🏆 Performance par ligue :")
        league_perf = df.groupby('league')['correct'].agg(['mean', 'count'])
        league_perf = league_perf.sort_values('mean', ascending=False)
        league_perf.columns = ['Accuracy', 'N_matchs']
        print(league_perf.to_string())
    
    # 6. Analyse selon ELO diff
    if 'elo_diff' in df.columns:
        print(f"\n⚖️ Performance selon l'écart ELO :")
        df['elo_diff_cat'] = pd.cut(df['elo_diff'], 
                                    bins=[-np.inf, -100, -50, 50, 100, np.inf],
                                    labels=['Away >> Home', 'Away > Home', 'Équilibré', 
                                           'Home > Away', 'Home >> Away'])
        elo_perf = df.groupby('elo_diff_cat')['correct'].agg(['mean', 'count'])
        elo_perf.columns = ['Accuracy', 'N_matchs']
        print(elo_perf.to_string())
    
    # 7. Pires erreurs (haute confiance mais mauvaise prédiction)
    print(f"\n❌ Top 10 pires prédictions (haute confiance, erreur) :")
    worst_errors = df[df['correct'] == 0].nlargest(10, 'max_prob')
    
    for idx, row in worst_errors.iterrows():
        real_label = {1: 'Home', 0: 'Draw', -1: 'Away'}[row['y_true']]
        pred_label = {1: 'Home', 0: 'Draw', -1: 'Away'}[row['y_pred']]
        print(f"   • {row.get('date', 'N/A')} | {row.get('home_team', '?')} vs {row.get('away_team', '?')}")
        print(f"     Réel: {real_label}, Prédit: {pred_label} (confiance: {row['max_prob']:.2%})")
    
    return df


def plot_error_analysis(df: pd.DataFrame, with_xg: bool = False):
    """
    Crée des visualisations de l'analyse d'erreur.
    
    Args:
        df: DataFrame avec les prédictions et résultats
        with_xg: Si True, sauvegarde dans le dossier xg
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution des probabilités prédites (correct vs incorrect)
    ax = axes[0, 0]
    df_correct = df[df['correct'] == 1]['max_prob']
    df_incorrect = df[df['correct'] == 0]['max_prob']
    
    ax.hist(df_correct, bins=30, alpha=0.6, label='Correct', color='green')
    ax.hist(df_incorrect, bins=30, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Probabilité maximale')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de la confiance du modèle')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Accuracy par confiance
    ax = axes[0, 1]
    df['confidence_bin'] = pd.cut(df['max_prob'], bins=10)
    conf_acc = df.groupby('confidence_bin')['correct'].agg(['mean', 'count'])
    
    bin_centers = [interval.mid for interval in conf_acc.index]
    ax.plot(bin_centers, conf_acc['mean'], marker='o', linewidth=2)
    ax.axhline(y=df['correct'].mean(), color='red', linestyle='--', 
               label='Accuracy globale')
    ax.set_xlabel('Confiance du modèle')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy selon la confiance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Matrice de confusion (heatmap)
    ax = axes[1, 0]
    confusion = pd.crosstab(
        df['y_true'].map({1: 'Home', 0: 'Draw', -1: 'Away'}),
        df['y_pred'].map({1: 'Home', 0: 'Draw', -1: 'Away'})
    )
    sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title('Matrice de confusion')
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')
    
    # 4. Accuracy par ligue
    if 'league' in df.columns:
        ax = axes[1, 1]
        league_acc = df.groupby('league')['correct'].mean().sort_values(ascending=True)
        league_acc.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Accuracy')
        ax.set_title('Performance par ligue')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = SavePaths.get_result_path(
        category='production',
        filename='error_analysis_plot.png',
        with_xg=with_xg
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique d'analyse d'erreur sauvegardé : {plot_path}")
    plt.close()


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              ANALYSE D'ERREUR DU MODÈLE XGBOOST               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Choix : avec ou sans XG
    with_xg = False  # 🔹 Change selon tes besoins
    
    # 1. Charger le meilleur modèle
    model_data = load_best_model(with_xg=with_xg)
    if model_data is None:
        return
    
    # 2. Charger les données
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    
    # 3. Faire les prédictions sur le test set
    features = model_data['features']
    model = model_data['model']
    
    X_test = test_df[features].values
    y_test = test_df[cfg.TARGET_COL].values
    probs_test = model.predict_proba(X_test)
    
    # 4. Analyser les erreurs
    test_analyzed = analyze_prediction_errors(
        test_df, y_test, probs_test, "Test Set"
    )
    
    # 5. Créer les visualisations
    plot_error_analysis(test_analyzed, with_xg)
    
    # 6. Sauvegarder le dataset avec prédictions
    result_path = SavePaths.get_result_path(
        category='production',
        filename='test_set_with_predictions.csv',
        with_xg=with_xg
    )
    test_analyzed.to_csv(result_path, index=False)
    print(f"\n✅ Dataset avec prédictions sauvegardé : {result_path}")
    
    print(f"\n{'='*70}")
    print("  ANALYSE TERMINÉE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
