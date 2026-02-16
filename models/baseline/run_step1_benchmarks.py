"""
STEP 1 — OFFICIAL BENCHMARK PIPELINE

Objectif :
- Établir un benchmark clair et irréfutable
- Comparer TOUS les modèles sur les MÊMES données
- Définir le seuil minimum à battre pour aller plus loin

Ce script est la BASE du projet.
"""

import numpy as np
import pandas as pd


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as configs
import models.utils as utils


from baseline_bookmaker import BookmakerBaseline
from baseline_simple import SimpleEloBaseline
from baseline_logreg import LogisticRegressionBaseline


# ========================================
# STEP 1.1 — LOAD & SPLIT DATA
# ========================================

print("\n STEP 1 — BENCHMARK DES MODÈLES\n")

df = utils.load_data(with_xg=False, merge_odds=True)
train_df, cv_df, test_df = utils.train_cv_test_split(df)

datasets = {
    "TRAIN": train_df,
    "CV": cv_df,
    "TEST": test_df
}


# ========================================
# STEP 1.2 — INIT MODELS
# ========================================

models = {
    "Bookmaker": BookmakerBaseline(),
    "Simple Elo": SimpleEloBaseline(home_advantage=100),
    "LogReg": LogisticRegressionBaseline()
}


# ========================================
# STEP 1.3 — TRAIN MODELS (SI NÉCESSAIRE)
# ========================================

print("\n ENTRAÎNEMENT DES MODÈLES\n")

models["LogReg"].fit(train_df)
# Bookmaker & Simple Elo n'ont pas besoin de fit


# ========================================
# STEP 1.4 — EVALUATION
# ========================================

all_results = []

for dataset_name, dataset in datasets.items():
    print(f"\n{'='*70}")
    print(f" ÉVALUATION — {dataset_name}")
    print(f"{'='*70}")

    y_true = dataset[configs.TARGET_COL].values

    for model_name, model in models.items():
        print(f"\n Modèle : {model_name}")

        y_pred_proba = model.predict_proba(dataset)

        results = utils.evaluate_predictions(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            odds_df=dataset,
            model_name=model_name
        )

        utils.print_evaluation_summary(results, dataset_name)
        all_results.append({**results, "dataset": dataset_name})


# ========================================
# STEP 1.5 — SAVE RESULTS
# ========================================

results_df = pd.DataFrame(all_results)
utils.save_results(results_df.to_dict("records"), "baseline/step1_baselines_benchmark.csv")

print("\n✅ STEP 1 TERMINÉ — BENCHMARK OFFICIEL ÉTABLI")

