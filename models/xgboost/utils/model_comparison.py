"""
Comparaison globale de tous les modèles :
- Baselines (Simple ELO, LogReg, Bookmaker)
- XGBoost Baseline
- XGBoost Optimized
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
from models.configs.save_paths import SavePaths

sns.set_style("whitegrid")


def load_all_results(with_xg: bool = False):
    """Charge tous les résultats disponibles"""
    results_dir = cfg.RESULTS_WITH_XG_DIR if with_xg else cfg.RESULTS_NO_XG_DIR
    
    all_results = []
    
    # 1. Baselines (Step 1)
    baseline_file = SavePaths.get_result_path(
        category='step1_baselines',
        filename=f"baseline_comparison_{'xg' if with_xg else 'no_xg'}.csv",
        with_xg=with_xg
    )
    if baseline_file.exists():
        baselines = pd.read_csv(baseline_file)
        baselines['step'] = 'Step 1 - Baselines'
        all_results.append(baselines)
        print(f"✓ Baselines chargées ({len(baselines)} lignes)")
    
    # 2. XGBoost baseline (Step 2a)
    step2a_file = SavePaths.get_result_path(
        category='step2a_baseline',
        filename=f"baseline_comparison_{'xg' if with_xg else 'no_xg'}.csv",
        with_xg=with_xg
    )
    if step2a_file.exists():
        step2a = pd.read_csv(step2a_file)
        step2a['step'] = 'Step 2a - XGBoost Baseline'
        all_results.append(step2a)
        print(f"✓ Step 2a chargé ({len(step2a)} lignes)")
    
    # 3. Feature comparison (Step 2b - partie 1)
    step2b_features = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison.csv',
        with_xg=with_xg
    )
    if step2b_features.exists():
        features = pd.read_csv(step2b_features)
        features['step'] = 'Step 2b - Feature Selection'
        all_results.append(features)
        print(f"✓ Step 2b (features) chargé ({len(features)} lignes)")
    
    # 4. Modèle final optimisé (Step 2b - partie 2)
    step2b_final = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='final_results.csv',
        with_xg=with_xg
    )
    if step2b_final.exists():
        final = pd.read_csv(step2b_final)
        final['step'] = 'Step 2b - Optimized Model'
        all_results.append(final)
        print(f"✓ Step 2b (optimized) chargé ({len(final)} lignes)")
    
    if not all_results:
        print("❌ Aucun résultat trouvé. Exécute d'abord les scripts précédents.")
        return None
    
    # Combiner tous les résultats
    df = pd.concat(all_results, ignore_index=True)
    
    # Standardiser les noms de colonnes si nécessaire
    if 'model_name' in df.columns and 'model' not in df.columns:
        df['model'] = df['model_name']
    
    return df


def create_comparison_table(df: pd.DataFrame):
    """Crée un tableau comparatif des modèles sur le TEST set"""
    
    # Filtrer uniquement le TEST set
    test_df = df[df['dataset'] == 'test'].copy()
    
    # Colonnes essentielles
    columns = ['step', 'model', 'accuracy', 'log_loss', 'brier_score', 'roi']
    
    # Vérifier quelles colonnes existent
    available_cols = [col for col in columns if col in test_df.columns]
    
    # Créer le tableau
    comparison = test_df[available_cols].copy()
    
    # Trier par log_loss (meilleur = plus bas)
    if 'log_loss' in comparison.columns:
        comparison = comparison.sort_values('log_loss')
    
    return comparison


def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: Path):
    """Crée une visualisation complète de tous les modèles"""
    
    test_df = df[df['dataset'] == 'test'].copy()
    
    # Trier par log_loss
    test_df = test_df.sort_values('log_loss')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Log Loss
    ax = axes[0, 0]
    bars = ax.barh(range(len(test_df)), test_df['log_loss'])
    
    # Colorer selon l'étape
    colors = {'Step 1 - Baselines': 'lightblue', 
              'Step 2a - XGBoost Baseline': 'orange',
              'Step 2b - Feature Selection': 'lightgreen',
              'Step 2b - Optimized Model': 'darkgreen'}
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        bars[i].set_color(colors.get(row['step'], 'gray'))
    
    ax.set_yticks(range(len(test_df)))
    ax.set_yticklabels(test_df['model'], fontsize=9)
    ax.set_xlabel('Log Loss (↓ meilleur)', fontsize=11)
    ax.set_title('Comparaison Log Loss', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Annoter les valeurs
    for i, (idx, row) in enumerate(test_df.iterrows()):
        ax.text(row['log_loss'], i, f" {row['log_loss']:.4f}", 
               va='center', fontsize=8)
    
    # 2. Accuracy
    ax = axes[0, 1]
    test_df_acc = test_df.sort_values('accuracy', ascending=False)
    bars = ax.barh(range(len(test_df_acc)), test_df_acc['accuracy'])
    
    for i, (idx, row) in enumerate(test_df_acc.iterrows()):
        bars[i].set_color(colors.get(row['step'], 'gray'))
    
    ax.set_yticks(range(len(test_df_acc)))
    ax.set_yticklabels(test_df_acc['model'], fontsize=9)
    ax.set_xlabel('Accuracy (↑ meilleur)', fontsize=11)
    ax.set_title('Comparaison Accuracy', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(test_df_acc.iterrows()):
        ax.text(row['accuracy'], i, f" {row['accuracy']:.3f}", 
               va='center', fontsize=8)
    
    # 3. Brier Score
    ax = axes[1, 0]
    test_df_brier = test_df.sort_values('brier_score')
    bars = ax.barh(range(len(test_df_brier)), test_df_brier['brier_score'])
    
    for i, (idx, row) in enumerate(test_df_brier.iterrows()):
        bars[i].set_color(colors.get(row['step'], 'gray'))
    
    ax.set_yticks(range(len(test_df_brier)))
    ax.set_yticklabels(test_df_brier['model'], fontsize=9)
    ax.set_xlabel('Brier Score (↓ meilleur)', fontsize=11)
    ax.set_title('Comparaison Brier Score', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(test_df_brier.iterrows()):
        ax.text(row['brier_score'], i, f" {row['brier_score']:.4f}", 
               va='center', fontsize=8)
    
    # 4. ROI (si disponible)
    ax = axes[1, 1]
    test_df_roi = test_df[test_df['roi'].notna()].copy()
    
    if len(test_df_roi) > 0:
        test_df_roi = test_df_roi.sort_values('roi', ascending=False)
        bars = ax.barh(range(len(test_df_roi)), test_df_roi['roi'])
        
        # Colorer en vert si positif, rouge si négatif
        for i, (idx, row) in enumerate(test_df_roi.iterrows()):
            color = 'green' if row['roi'] > 0 else 'red'
            bars[i].set_color(color)
            bars[i].set_alpha(0.6)
        
        ax.set_yticks(range(len(test_df_roi)))
        ax.set_yticklabels(test_df_roi['model'], fontsize=9)
        ax.set_xlabel('ROI % (↑ meilleur)', fontsize=11)
        ax.set_title('Comparaison ROI', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(test_df_roi.iterrows()):
            ax.text(row['roi'], i, f" {row['roi']:.2f}%", 
                   va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'ROI non disponible', 
               ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=step) 
                      for step, color in colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.suptitle('COMPARAISON GLOBALE DES MODÈLES (Test Set)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # Sauvegarder
    output_file = output_dir / "global_model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique global sauvegardé : {output_file}")
    plt.close()


def display_improvement_summary(df: pd.DataFrame):
    """Affiche un résumé des améliorations"""
    
    test_df = df[df['dataset'] == 'test'].copy()
    
    print(f"\n{'='*70}")
    print("  RÉSUMÉ DES AMÉLIORATIONS")
    print(f"{'='*70}\n")
    
    # Identifier le modèle baseline et le meilleur modèle
    bookmaker_models = ["Bookmaker (Simple)", "Bookmaker (Proportional)"]
    if any(model in test_df['model'].values for model in bookmaker_models):
        
        baseline_df = test_df[test_df['model'].isin(bookmaker_models)]
        
        # On prend le meilleur bookmaker (plus petit log loss)
        baseline = baseline_df.sort_values("log_loss").iloc[0]
        baseline_name = baseline["model"]
    else:
        baseline = test_df.iloc[-1]  # Prendre le pire
        baseline_name = baseline['model']
        raise ValueError("❌ Aucun baseline bookmaker trouvé dans test_df.")
    
    # On exclut les bookmakers
    non_bookmaker_df = test_df[~test_df['model'].isin(bookmaker_models)]
    if non_bookmaker_df.empty:
        raise ValueError("❌ Aucun modèle ML disponible (seulement des bookmakers).")
    best = non_bookmaker_df.sort_values('log_loss').iloc[0]
    top2 = non_bookmaker_df.sort_values('log_loss').iloc[1]
    top3 = non_bookmaker_df.sort_values('log_loss').iloc[2]

    print(f"📍 BASELINE : {baseline_name}")
    print(f"   • Log Loss    : {baseline['log_loss']:.4f}")
    print(f"   • Brier Score : {baseline['brier_score']:.4f}")
    print(f"   • Accuracy    : {baseline['accuracy']:.4f}")
    
    print(f"\n🏆 MEILLEUR MODÈLE : {best['model']}")
    print(f"   • Log Loss    : {best['log_loss']:.4f}")
    print(f"   • Brier Score : {best['brier_score']:.4f}")
    print(f"   • Accuracy    : {best['accuracy']:.4f}")

    print(f"\n🏆 TOP2 MODÈLE : {top2['model']}")
    print(f"   • Log Loss    : {top2['log_loss']:.4f}")
    print(f"   • Brier Score : {top2['brier_score']:.4f}")
    print(f"   • Accuracy    : {top2['accuracy']:.4f}")

    print(f"\n🏆 TOP3 MODÈLE : {top3['model']}")
    print(f"   • Log Loss    : {top3['log_loss']:.4f}")
    print(f"   • Brier Score : {top3['brier_score']:.4f}")
    print(f"   • Accuracy    : {top3['accuracy']:.4f}")
    
    print(f"\n📈 AMÉLIORATION :")
    log_improv = (baseline['log_loss'] - best['log_loss']) / baseline['log_loss'] * 100
    brier_improv = (baseline['brier_score'] - best['brier_score']) / baseline['brier_score'] * 100
    acc_improv = (best['accuracy'] - baseline['accuracy']) * 100
    
    print(f"   • Log Loss    : {log_improv:+.2f}%")
    print(f"   • Brier Score : {brier_improv:+.2f}%")
    print(f"   • Accuracy    : {acc_improv:+.2f} points")
    
    if pd.notna(best['roi']) and pd.notna(baseline.get('roi')):
        roi_improv = best['roi'] - baseline['roi']
        print(f"   • ROI         : {roi_improv:+.2f} points")
    
    print(f"\n{'='*70}\n")


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            COMPARAISON GLOBALE DE TOUS LES MODÈLES            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Choix : avec ou sans XG
    with_xg = False  #  Change selon tes besoins
    
    # 1. Charger tous les résultats
    print("\n Chargement des résultats...")
    df = load_all_results(with_xg=with_xg)
    
    if df is None:
        return
    
    print(f"\n✅ {len(df)} résultats chargés au total")
    print(f"   • Étapes : {df['step'].nunique()}")
    print(f"   • Modèles : {df['model'].nunique()}")
    print(f"   • Datasets : {', '.join(df['dataset'].unique())}")
    
    # 2. Créer le tableau comparatif
    print(f"\n{'='*70}")
    print("  TABLEAU COMPARATIF (TEST SET)")
    print(f"{'='*70}\n")
    
    comparison = create_comparison_table(df)
    print(comparison.to_string(index=False))
    
    # 3. Résumé des améliorations
    display_improvement_summary(df)
    
    # 4. Créer les visualisations
    results_dir = cfg.RESULTS_WITH_XG_DIR if with_xg else cfg.RESULTS_NO_XG_DIR
    plot_comprehensive_comparison(df, results_dir)
    
    # 5. Sauvegarder le tableau comparatif
    output_file = results_dir / "global_comparison_all_models.csv"
    comparison.to_csv(output_file, index=False)
    print(f"✅ Tableau comparatif sauvegardé : {output_file}")
    
    print(f"\n{'='*70}")
    print("  COMPARAISON GLOBALE TERMINÉE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
