"""
run_predictions.py — Prédictions matchs futurs (2-3x par semaine)

Ordre :
  1. Récupération cotes futures (Odds API)
  2. Normalisation noms équipes
  3. Merge matchs + cotes
  4. Prédictions + Value Bets (2 stratégies)

Fichiers générés (écrasés à chaque lancement) :
  results/predictions/predictions_full.csv
  results/predictions/value_bets_volume.csv
  results/predictions/value_bets_pure.csv

Usage :
    python run_predictions.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE   = Path(__file__).parent
PYTHON = sys.executable

SCRIPTS = [
    ("1/4  Récupération cotes futures",
     BASE / "src/scraping/odds_futur/fetch_upcoming_odds.py"),

    ("2/4  Normalisation noms équipes",
     BASE / "src/team_mapping.py"),

    ("3/4  Merge matchs + cotes",
     BASE / "src/create_futur_match_odds.py"),

    ("4/4  Prédictions + Value Bets",
     BASE / "src/predict_future_matches.py"),
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
        print(f"  ✓ Terminé en {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ Échoué (code {result.returncode})")
        return False


def main():
    print("=" * 65)
    print("  PIPELINE PRÉDICTIONS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    results = {}
    for label, script in SCRIPTS:
        ok = run(label, script)
        results[label] = ok
        if not ok:
            print(f"  ⚠ Étape échouée — continuation quand même...")

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ")
    print(f"{'='*65}")
    n_ok = sum(results.values())
    for label, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {label}")
    print(f"\n  {n_ok}/{len(results)} étapes réussies")
    if n_ok == len(results):
        print("\n  Résultats dans results/predictions/ :")
        print("    • predictions_full.csv")
        print("    • value_bets_volume.csv  (ROI +8%, ~110 paris/an)")
        print("    • value_bets_pure.csv    (ROI +16%, très sélectif)")


if __name__ == "__main__":
    main()