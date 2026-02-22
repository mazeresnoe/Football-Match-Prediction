"""
Visualisation et analyse des résultats d'amélioration du XGBoost
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
from models.configs.save_paths import SavePaths

# Configuration style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_feature_comparison(results_file: Path, with_xg: bool = False):
    """
    Compare les performances selon le nombre de features
    
    Args:
        results_file: Chemin du fichier de résultats
        with_xg: Si True, sauvegarde dans le dossier xg
    """
    df = pd.read_csv(results_file)
    test_df = df[df['dataset'] == 'test'].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['log_loss', 'accuracy', 'roi']
    titles = ['Log Loss (↓ meilleur)', 'Accuracy (↑ meilleur)', 'ROI % (↑ meilleur)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Trier par métrique
        if metric in ['accuracy', 'roi']:
            sorted_df = test_df.sort_values(metric, ascending=False)
        else:
            sorted_df = test_df.sort_values(metric, ascending=True)
        
        # Plot
        bars = ax.barh(sorted_df['model'], sorted_df[metric])
        
        # Colorer le meilleur
        bars[0].set_color('green')
        bars[0].set_alpha(0.7)
        
        ax.set_xlabel(title)
        ax.set_title(f"{title}")
        ax.grid(axis='x', alpha=0.3)
        
        # Annoter les valeurs
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            value = row[metric]
            if pd.notna(value):
                ax.text(value, i, f" {value:.3f}", 
                       va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison_plot.png',
        with_xg=with_xg
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {plot_path}")
    plt.close()


def plot_metrics_evolution(results_file: Path, with_xg: bool = False):
    """
    Montre l'évolution des métriques sur TRAIN/CV/TEST
    
    Args:
        results_file: Chemin du fichier de résultats
        with_xg: Si True, sauvegarde dans le dossier xg
    """
    df = pd.read_csv(results_file)
    
    models = df['model'].unique()
    datasets = ['train', 'cv', 'test']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['log_loss', 'accuracy', 'brier_score']
    titles = ['Log Loss', 'Accuracy', 'Brier Score']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for model in models:
            model_data = df[df['model'] == model]
            values = [model_data[model_data['dataset'] == d][metric].values[0] 
                     for d in datasets]
            ax.plot(datasets, values, marker='o', label=model, linewidth=2)
        
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle("Évolution des métriques (Train → CV → Test)", 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='metrics_evolution_plot.png',
        with_xg=with_xg
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {plot_path}")
    plt.close()


def generate_summary_report(results_dir: Path, with_xg: bool = False):
    """Génère un rapport complet des résultats"""
    
    print(f"\n{'='*70}")
    print(f"  RAPPORT RÉCAPITULATIF - {'AVEC' if with_xg else 'SANS'} XG")
    print(f"{'='*70}\n")
    
    # Charger les résultats de feature comparison
    feature_comp_file = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison.csv',
        with_xg=with_xg
    )

    if not feature_comp_file.exists():
        print(f"❌ Fichier introuvable : {feature_comp_file}")
        print(f"   Exécute d'abord 'step2b_optimization.py'")
        return
    
    df = pd.read_csv(feature_comp_file)
    
    # Résultats sur le TEST set
    test_df = df[df['dataset'] == 'test'].sort_values('log_loss')
    
    print("📊 CLASSEMENT DES MODÈLES (sur TEST set)")
    print("-" * 70)
    print(test_df[['model', 'accuracy', 'log_loss', 
                   'brier_score', 'roi']].to_string(index=False))
    
    # Meilleur modèle
    best_model = test_df.iloc[0]
    print(f"\n🏆 MEILLEUR MODÈLE : {best_model['model']}")
    print(f"   • Accuracy           : {best_model['accuracy']:.4f}")
    print(f"   • Log Loss           : {best_model['log_loss']:.4f}")
    print(f"   • Brier Score        : {best_model['brier_score']:.4f}")
    if pd.notna(best_model['roi']):
        print(f"   • ROI                : {best_model['roi']:.2f}%")
        print(f"   • Profit             : ${best_model['profit']:.2f}")
        print(f"   • Nombre de paris    : {best_model['n_bets']:.0f}")
    
    # Comparaison avec baseline
    baseline_file = SavePaths.get_result_path(
        category='step2a_baseline',
        filename=f'baseline_comparison_{"xg" if with_xg else "no_xg"}.csv',
        with_xg=with_xg
    )
    
    if baseline_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        baseline_test = baseline_df[baseline_df['dataset'] == 'test']
        
        print(f"\n📈 AMÉLIORATION vs BASELINE XGBOOST")
        print("-" * 70)
        
        if 'XGBoost' in baseline_test['model'].values:
            baseline_xgb = baseline_test[baseline_test['model'] == 'XGBoost'].iloc[0]
            
            acc_improv = (best_model['accuracy'] - baseline_xgb['accuracy']) * 100
            log_improv = (baseline_xgb['log_loss'] - best_model['log_loss']) / baseline_xgb['log_loss'] * 100
            brier_improv = (baseline_xgb['brier_score'] - best_model['brier_score']) / baseline_xgb['brier_score'] * 100
            
            print(f"   • Accuracy    : {acc_improv:+.2f} points")
            print(f"   • Log Loss    : {log_improv:+.2f}% (amélioration)")
            print(f"   • Brier Score : {brier_improv:+.2f}% (amélioration)")
            
            if pd.notna(best_model['roi']) and pd.notna(baseline_xgb['roi']):
                roi_improv = best_model['roi'] - baseline_xgb['roi']
                print(f"   • ROI         : {roi_improv:+.2f} points")
    
    # Vérifier si le modèle final existe
    final_results_file = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='final_results.csv',
        with_xg=with_xg
    )
    if final_results_file.exists():
        print(f"\n✅ Résultats du modèle optimisé disponibles dans :")
        print(f"   {final_results_file}")
        
        final_df = pd.read_csv(final_results_file)
        final_test = final_df[final_df['dataset'] == 'test'].iloc[0]
        
        print(f"\n🎯 MODÈLE FINAL OPTIMISÉ (avec hyperparameter tuning)")
        print("-" * 70)
        print(f"   • Accuracy           : {final_test['accuracy']:.4f}")
        print(f"   • Log Loss           : {final_test['log_loss']:.4f}")
        print(f"   • Brier Score        : {final_test['brier_score']:.4f}")
        if pd.notna(final_test['roi']):
            print(f"   • ROI                : {final_test['roi']:.2f}%")
    
    print(f"\n{'='*70}\n")


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        VISUALISATION ET ANALYSE DES RÉSULTATS                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Choix : avec ou sans XG
    with_xg = False  # 🔹 Change selon tes besoins
    
    results_dir = cfg.RESULTS_WITH_XG_DIR if with_xg else cfg.RESULTS_NO_XG_DIR
    
    # 1. Générer le rapport
    generate_summary_report(results_dir, with_xg=with_xg)
    
    # 2. Créer les visualisations
    feature_comp_file = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison.csv',
        with_xg=with_xg
    )
    
    if feature_comp_file.exists():
        print("\n📊 Génération des graphiques...")
        plot_feature_comparison(feature_comp_file, with_xg)
        plot_metrics_evolution(feature_comp_file, with_xg)
        print("\n✅ Visualisations créées avec succès !")
    else:
        print(f"\n⚠️ Fichier de résultats introuvable : {feature_comp_file}")
        print("   Exécute d'abord 'xgboost_improved.py'")


if __name__ == "__main__":
    main()
