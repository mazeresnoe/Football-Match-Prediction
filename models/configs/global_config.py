"""
Configuration globale pour les modèles de prédiction football
"""

import os
from pathlib import Path

# ========================================
# CHEMINS DES FICHIERS
# ========================================
BASE_DIR = Path(__file__).parent.parent.parent  # remonte de 2 niveaux depuis configs/
DATA_DIR = BASE_DIR / "data"

# Données avec XG
DATA_WITH_XG = DATA_DIR / "clean/prematch/etape3/full_dataset_with_xg_clean_v2.csv"

# Données sans XG
DATA_NO_XG = DATA_DIR / "clean/prematch/etape3/full_dataset_no_xg_clean_v2.csv"

# Odds
DATA_ODDS = DATA_DIR / "odds/all_odds_standardized.csv"

# Dossiers de sortie
RESULTS_DIR = BASE_DIR / "results/modeling"
RESULTS_NO_XG_DIR = RESULTS_DIR / "no_xg"
RESULTS_WITH_XG_DIR = RESULTS_DIR / "xg"

RESULTS_NO_XG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_WITH_XG_DIR.mkdir(parents=True, exist_ok=True)

# Dossier pour sauvegarder les modèles
MODELS_DIR = BASE_DIR / "models/saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# PARAMÈTRES GÉNÉRAUX
# ========================================
TARGET_COL = "result"
RANDOM_STATE = 42

# Mapping des résultats
RESULT_MAPPING = {1: "Home", 0: "Draw", -1: "Away"}

# ========================================
# FEATURES
# ========================================

# Features de base (pour baseline simple - Elo uniquement)
BASIC_FEATURES = [
    "elo_home",
    "elo_away",
    "elo_diff",
]

# Features Logistic Regression (sans XG)
LOGREG_FEATURES_NO_XG = [
    "elo_home", "elo_away", "elo_diff", "elo_diff_squared",
    "home_form_5", "away_form_5", "diff_form_5",
    "home_goals_scored_avg_5", "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
    "home_goal_diff_avg_5", "away_goal_diff_avg_5",
    "home_form_10", "away_form_10",
    "home_goals_scored_avg_10", "away_goals_scored_avg_10",
    "home_goals_conceded_avg_10", "away_goals_conceded_avg_10",
    "h2h_home_wins_10", "h2h_away_wins_10", "h2h_draws_10",
    "rest_advantage",
]

# Features Logistic Regression (avec XG)
LOGREG_FEATURES_WITH_XG = LOGREG_FEATURES_NO_XG + [
    "home_expectedgoals_avg_5", "away_expectedgoals_avg_5",
    "home_expectedgoals_conceded_avg_5", "away_expectedgoals_conceded_avg_5",
    "diff_xg_5", "home_xg_overperf_10", "away_xg_overperf_10",
]

# ========================================
# SPLIT TRAIN/CV/TEST
# ========================================
TRAIN_PCT = 0.60
CV_PCT = 0.20
TEST_PCT = 0.20

# ========================================
# MÉTRIQUES D'ÉVALUATION / PARIS
# ========================================
MIN_CONFIDENCE_THRESHOLD = 0.40
INITIAL_BANKROLL = 1000
STAKE_PER_MATCH = 10

# ========================================
# PARAMÈTRES XGBOOST
# ========================================
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "mlogloss",
}

# ========================================
# AFFICHAGE
# ========================================
VERBOSE = 1

if VERBOSE >= 1:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     CONFIGURATION CHARGÉE - Football Prediction Pipeline      ║
╚══════════════════════════════════════════════════════════════╝
 Chemins :
   • Base directory    : {BASE_DIR}
   • Data directory    : {DATA_DIR}
   • Avec XG           : {'✓' if DATA_WITH_XG.exists() else '✗ MANQUANT'}
   • Sans XG           : {'✓' if DATA_NO_XG.exists() else '✗ MANQUANT'}  
   • Odds              : {'✓' if DATA_ODDS.exists() else '✗ MANQUANT'}
 Résultats :
   • Sans XG           : {RESULTS_NO_XG_DIR}
   • Avec XG           : {RESULTS_WITH_XG_DIR}
 Target : {TARGET_COL}, Seed : {RANDOM_STATE}
 Split temporel : TRAIN {TRAIN_PCT*100:.0f}%, CV {CV_PCT*100:.0f}%, TEST {TEST_PCT*100:.0f}%
""")
