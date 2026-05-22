"""
run_model_training.py — Pipeline complet d'entraînement (toutes les 2 semaines)

Ordre :
  1.  Baselines (bookmaker, logreg, simple)
  2.  Step 2a — XGBoost baseline
  3.  Step 2b — Optimisation hyperparamètres (optuna, ~1-2h)
  4.  Step 2c — Calibration
  5.  Step 3  — Stratégies (grid search value bets)
  6.  Value Bet Detector (validation sur test set)

Prérequis : run_weekly.py doit avoir tourné avant
(données à jour dans data/clean/prematch/etape3/)

Usage :
    python run_model_training.py
    python run_model_training.py --skip-baselines   # skip étapes 1-2a
    python run_model_training.py --skip-optuna      # skip step2b, utilise params existants
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE   = Path(__file__).parent
PYTHON = sys.executable

SCRIPTS_BASELINES = [
    ("1/6  Baselines (bookmaker + logreg + simple)",
     BASE / "models/baseline/run_step1_benchmarks.py"),

    ("2/6  XGBoost baseline (step 2a)",
     BASE / "models/xgboost/core/step2a_baseline.py"),
]

SCRIPTS_MODEL = [
    ("3/6  Optimisation hyperparamètres (step 2b — optuna)",
     BASE / "models/xgboost/core/step2b_optimization.py"),

    ("4/6  Calibration (step 2c)",
     BASE / "models/xgboost/core/step2c_calibration.py"),

    ("5/6  Stratégies value bets (step 3 — grid search)",
     BASE / "models/xgboost/core/step3_strategies.py"),

    ("6/6  Value Bet Detector (validation test set)",
     BASE / "models/xgboost/core/value_bet_detector.py"),
]


def run(label: str, script: Path) -> bool:
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"  → {script.name}")
    print(f"{'─'*65}")

    if not script.exists():
        print(f"  ⚠ Script introuvable : {script}")
        print(f"  → Étape ignorée")
        return False

    start = time.time()
    result = subprocess.run([PYTHON, str(script)], cwd=str(BASE))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ Terminé en {elapsed:.0f}s ({elapsed/60:.1f}min)")
        return True
    else:
        print(f"  ✗ Échoué (code {result.returncode})")
        return False


def main():
    skip_baselines = "--skip-baselines" in sys.argv
    skip_optuna    = "--skip-optuna"    in sys.argv

    print("=" * 65)
    print("  PIPELINE ENTRAÎNEMENT MODÈLE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if skip_baselines:
        print("  Mode : sans baselines")
    if skip_optuna:
        print("  Mode : sans optuna (params existants)")
    print("=" * 65)

    start_total = time.time()
    results = {}

    # Baselines
    if not skip_baselines:
        for label, script in SCRIPTS_BASELINES:
            ok = run(label, script)
            results[label] = ok
            if not ok:
                print(f"  ⚠ Étape échouée — continuation quand même...")
    else:
        print("\n  ⏭ Baselines ignorées (--skip-baselines)")

    # Modèle principal
    for label, script in SCRIPTS_MODEL:
        # Skip optuna si demandé
        if skip_optuna and "step2b" in str(script):
            print(f"\n  ⏭ Ignoré (--skip-optuna) : {script.name}")
            print(f"  → Le modèle existant sera utilisé pour la calibration")
            continue

        ok = run(label, script)
        results[label] = ok
        if not ok:
            print(f"  ⚠ Étape échouée — continuation quand même...")

    elapsed_total = time.time() - start_total

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ — {elapsed_total/60:.0f} minutes au total")
    print(f"{'='*65}")
    n_ok = sum(results.values())
    for label, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {label}")
    print(f"\n  {n_ok}/{len(results)} étapes réussies")

    if n_ok == len(results):
        print("\n  Modèle mis à jour ✓")
        print("  Modèle sauvegardé dans : models/saved/production/")
        print("  Stratégies dans        : results/modeling/no_xg/production/strategy/")
        print("\n  → Lance run_predictions.py pour les nouvelles prédictions")
    else:
        n_fail = len(results) - n_ok
        print(f"\n  ⚠ {n_fail} étape(s) échouée(s) — vérifie les logs")


if __name__ == "__main__":
    main()