"""
run_weekly.py — Pipeline hebdomadaire (1x par semaine après journée de matchs)

Ordre :
  1. Scraping saison en cours (event_ids)
  2. Scraping stats des matchs
  3. Scraping odds historiques
  4. Preprocessing étape 1 (merge + cleaning)
  5. Feature engineering étape 1 (basic + advanced)
  6. Preprocessing étape 2 (split + EDA + cleaning + recovery)

Usage :
    python run_weekly.py
    python run_weekly.py --skip-scraping   # si données déjà scrappées
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE   = Path(__file__).parent
PYTHON = sys.executable

SCRIPTS = [
    # ── Scraping ──────────────────────────────────────────────
    ("1/6  Scraping event_ids saison en cours",
     BASE / "src/scraping/sofascore_scrap/round_scrapper_2.py"),

    ("2/6  Scraping stats des matchs",
     BASE / "src/scraping/sofascore_scrap/match_stats_scraper.py"),

    ("3/6  Scraping odds historiques",
     BASE / "src/scraping/odds/scrape_odds.py"),

    # ── Preprocessing étape 1 ────────────────────────────────
    ("4/6  Merge + Cleaning étape 1",
     BASE / "preprocessing/1_etape/1_raw_merger.py"),

    ("4/6  Cleaning étape 1",
     BASE / "preprocessing/1_etape/2_cleaning.py"),

    # ── Feature Engineering ───────────────────────────────────
    ("5/6  Feature Engineering V1 (basic)",
     BASE / "src/feature_engineering/etape1/3_feature_eng_v1_basic.py"),

    ("5/6  Feature Engineering V2 (advanced)",
     BASE / "src/feature_engineering/etape1/4_feature_eng_v2_advanced.py"),

    # ── Preprocessing étape 2 ────────────────────────────────
    ("6/6  Split dual dataset",
     BASE / "preprocessing/etape2/5_dual_dataset_splitter_v2.py"),

    ("6/6  EDA",
     BASE / "preprocessing/etape2/6_eda/6_eda2_comprehensive_analysis.py"),

    ("6/6  Cleaning étape 2",
     BASE / "preprocessing/etape2/7_cleaning.py"),

    ("6/6  Feature recovery",
     BASE / "preprocessing/etape2/8_feature_recovery.py"),
]

SCRAPING_SCRIPTS = {
    BASE / "src/scraping/sofascore_scrap/round_scrapper_2.py",
    BASE / "src/scraping/sofascore_scrap/match_stats_scraper.py",
    BASE / "src/scraping/odds/scrape_odds.py",
}


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
    skip_scraping = "--skip-scraping" in sys.argv

    print("=" * 65)
    print("  PIPELINE HEBDOMADAIRE — MISE À JOUR HISTORIQUE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if skip_scraping:
        print("  Mode : sans scraping")
    print("=" * 65)

    results = {}
    for label, script in SCRIPTS:
        if skip_scraping and script in SCRAPING_SCRIPTS:
            print(f"\n  ⏭ Ignoré (--skip-scraping) : {script.name}")
            continue
        ok = run(label, script)
        results[label] = ok
        if not ok:
            print(f"\n  ⚠ Étape échouée — continuation quand même...")

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ")
    print(f"{'='*65}")
    n_ok = sum(results.values())
    for label, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {label}")
    print(f"\n  {n_ok}/{len(results)} étapes réussies")
    if n_ok == len(results):
        print("\n  Pipeline hebdomadaire complet ✓")
        print("  → Lance run_predictions.py pour les prédictions de la semaine")


if __name__ == "__main__":
    main()