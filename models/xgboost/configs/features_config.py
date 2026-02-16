"""
Configuration des features pour XGBoost - Version enrichie
Exploite toutes les features déjà calculées dans le feature engineering
"""

# ========================================
# FEATURES XGBOOST ENRICHIES (SANS XG)
# ========================================

XGBOOST_FEATURES_NO_XG_V1 = [
    # ===== ELO =====
    "elo_home", "elo_away", "elo_diff", 
    "elo_diff_squared", "elo_diff_abs", "elo_diff_log",
    
    # ===== FORME RÉCENTE (5 matchs) =====
    "home_form_5", "away_form_5", "diff_form_5",
    "home_form_volatility", "away_form_volatility", "diff_form_volatility",
    
    # ===== FORME ÉTENDUE (10 matchs) =====
    "home_form_10", "away_form_10",
    "home_win_rate_10",
    
    # ===== MOMENTUM =====
    "home_momentum_form", "away_momentum_form", "diff_momentum_form",
    "home_momentum_goals", "away_momentum_goals", "diff_momentum_goals",
    "home_momentum_defense", "away_momentum_defense", "diff_momentum_defense",
    
    # ===== BUTS (5 matchs) =====
    "home_goals_scored_avg_5", "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
    "home_goal_diff_avg_5", "away_goal_diff_avg_5",
    "diff_goals_conceded_avg_5",
    "home_goals_volatility", "away_goals_volatility",
    
    # ===== BUTS (10 matchs) =====
    "home_goals_scored_avg_10", "away_goals_scored_avg_10",
    "home_goals_conceded_avg_10", "away_goals_conceded_avg_10",
    "home_goal_diff_10", "away_goal_diff_10", "diff_goal_diff_10",
    "diff_goals_conceded_avg_10",
    "home_clean_sheet_rate_10",
    
    # ===== TIRS & CONVERSION =====
    "home_shotsongoal_avg_5", "away_shotsongoal_avg_5",
    "home_shotsongoal_avg_10", "away_shotsongoal_avg_10",
    "home_shot_conversion_5", "away_shot_conversion_5", "diff_shot_conversion_5",
    "home_totalshotsinsidebox_avg_5", "away_totalshotsinsidebox_avg_5",
    "home_shotsoffgoal_avg_5", "away_shotsoffgoal_avg_5",
    
    # ===== GROSSES OCCASIONS =====
    "home_bigchancecreated_avg_5", "away_bigchancecreated_avg_5",
    "home_bigchancemissed_avg_5", "away_bigchancemissed_avg_5",
    "home_bigchancecreated_avg_10", "away_bigchancecreated_avg_10",
    "diff_bigchance_created_10",
    "home_bigchance_conversion_10", "away_bigchance_conversion_10", "diff_bigchance_conversion",
    
    # ===== CORNERS =====
    "home_cornerkicks_avg_5", "away_cornerkicks_avg_5",
    "home_cornerkicks_avg_10", "away_cornerkicks_avg_10",
    "diff_corners_5", "diff_corners_10",
    
    # ===== POSSESSION & PASSES =====
    "home_ballpossession_conceded_avg_5", "away_ballpossession_conceded_avg_5",
    "home_accuratepasses_conceded_avg_5", "away_accuratepasses_conceded_avg_5",
    "home_accuratepasses_avg_10", "away_accuratepasses_avg_10",
    
    # ===== ZONE OFFENSIVE =====
    "home_finalthirdentries_avg_5", "away_finalthirdentries_avg_5",
    "home_finalthirdentries_avg_10", "away_finalthirdentries_avg_10",
    "diff_finalthird_10",
    
    # ===== DÉFENSE =====
    "home_goalkeepersaves_avg_5", "away_goalkeepersaves_avg_5",
    "home_interceptionwon_avg_5", "away_interceptionwon_avg_5",
    "home_totalclearance_avg_5", "away_totalclearance_avg_5",
    "home_blockedscoringattempt_avg_5", "away_blockedscoringattempt_avg_5",
    "home_defense_efficiency_10", "away_defense_efficiency_10", "diff_defense_efficiency",
    
    # ===== DUELS & TACKLES =====
    "home_totaltackle_avg_5", "away_totaltackle_avg_5",
    "home_wontacklepercent_avg_5", "away_wontacklepercent_avg_5",
    "home_groundduelspercentage_avg_5", "away_groundduelspercentage_avg_5",
    "home_duelwonpercent_conceded_avg_5", "away_duelwonpercent_conceded_avg_5",
    "home_aerialduelspercentage_conceded_avg_5",
    
    # ===== DISCIPLINE =====
    "home_yellowcards_avg_5", "away_yellowcards_avg_5",
    "home_offsides_avg_5", "away_offsides_avg_5",
    
    # ===== RATIOS ATTAQUE/DÉFENSE =====
    "home_attack_defense_ratio", "away_attack_defense_ratio", "diff_attack_defense_ratio",
    
    # ===== HEAD TO HEAD =====
    "h2h_home_wins_10", "h2h_draws_10", "h2h_away_wins_10",
    "h2h_goals_for_10", "h2h_goals_against_10",
    "h2h_home_winrate_10", "h2h_momentum", "h2h_goal_diff_10",
    
    # ===== REPOS & CONTEXTE =====
    "days_rest_home", "days_rest_away", "rest_advantage",
    "month", "day_of_week", "season_month",
    
    # ===== INTERACTIONS ELO =====
    "elo_x_form_5", "elo_x_form_10",
    "elo_x_goals_5", "elo_x_goals_10",
    "elo_x_rest",
    
    # ===== INTERACTIONS LOG =====
    "rest_advantage_log", "diff_goal_diff_10_log",
    "elo_x_form_5_log", "elo_x_form_10_log",
    "elo_x_goals_5_log", "elo_x_goals_10_log",
    "elo_x_rest_log", "elo_diff_squared_log",
    "home_attack_defense_ratio_log", "away_attack_defense_ratio_log",
    "diff_attack_defense_ratio_log",
]

# ========================================
# FEATURES XGBOOST ENRICHIES (AVEC XG)
# ========================================

XGBOOST_FEATURES_WITH_XG_V1 = XGBOOST_FEATURES_NO_XG_V1 + [
    # ===== XG BASIQUE =====
    "home_expectedgoals_avg_5", "away_expectedgoals_avg_5",
    "home_expectedgoals_conceded_avg_5", "away_expectedgoals_conceded_avg_5",
    "home_expectedgoals_avg_10", "away_expectedgoals_avg_10",
    "home_expectedgoals_conceded_avg_10", "away_expectedgoals_conceded_avg_10",
    
    # ===== XG AVANCÉ =====
    "diff_xg_5", "diff_xg_10",
    "home_xg_overperf_10", "away_xg_overperf_10", "diff_xg_overperf",
    "xg_momentum",
    
    # ===== H2H XG =====
    "h2h_xg_for_10", "h2h_xg_against_10",
    
    # ===== AUTRES STATS DISPONIBLES AVEC XG =====
    "home_touchesinoppbox_avg_5", "away_touchesinoppbox_avg_5",
    "home_goalsprevented_avg_5", "away_goalsprevented_avg_5",
    "home_goalsprevented_avg_10", "away_goalsprevented_avg_10",
]

# ========================================
# FEATURES PAR CATÉGORIE (pour analyse)
# ========================================

FEATURE_CATEGORIES = {
    "elo": ["elo_home", "elo_away", "elo_diff", "elo_diff_squared", "elo_diff_abs", "elo_diff_log"],
    "form": ["home_form_5", "away_form_5", "diff_form_5", "home_form_10", "away_form_10", 
             "home_form_volatility", "away_form_volatility"],
    "goals": ["home_goals_scored_avg_5", "away_goals_scored_avg_5", "home_goals_conceded_avg_5",
              "away_goals_conceded_avg_5", "home_goal_diff_10", "away_goal_diff_10"],
    "shots": ["home_shotsongoal_avg_5", "away_shotsongoal_avg_5", "home_shot_conversion_5",
              "away_shot_conversion_5"],
    "defense": ["home_defense_efficiency_10", "away_defense_efficiency_10", "home_clean_sheet_rate_10"],
    "h2h": ["h2h_home_wins_10", "h2h_draws_10", "h2h_away_wins_10", "h2h_momentum"],
    "context": ["rest_advantage", "month", "day_of_week"],
    "xg": ["home_expectedgoals_avg_5", "away_expectedgoals_avg_5", "diff_xg_5",
           "home_xg_overperf_10", "away_xg_overperf_10"],
}

# ========================================
# VERSIONS PROGRESSIVES (pour tests)
# ========================================

# Version minimaliste (baseline actuelle)
XGBOOST_FEATURES_MINIMAL = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_5", "away_form_5", "diff_form_5",
    "home_goals_scored_avg_5", "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
    "home_form_10", "away_form_10",
]

# Version moyenne (50 features)
XGBOOST_FEATURES_MEDIUM = [
    # ELO
    "elo_home", "elo_away", "elo_diff", "elo_diff_squared",
    # Forme
    "home_form_5", "away_form_5", "diff_form_5", "home_form_10", "away_form_10",
    "home_momentum_form", "away_momentum_form",
    # Buts
    "home_goals_scored_avg_5", "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
    "home_goals_scored_avg_10", "away_goals_scored_avg_10",
    "home_goal_diff_10", "away_goal_diff_10", "diff_goal_diff_10",
    # Tirs
    "home_shotsongoal_avg_5", "away_shotsongoal_avg_5",
    "home_shot_conversion_5", "away_shot_conversion_5",
    "home_bigchancecreated_avg_5", "away_bigchancecreated_avg_5",
    # Défense
    "home_defense_efficiency_10", "away_defense_efficiency_10",
    "home_clean_sheet_rate_10",
    # Ratios
    "home_attack_defense_ratio", "away_attack_defense_ratio",
    # H2H
    "h2h_home_wins_10", "h2h_draws_10", "h2h_away_wins_10",
    "h2h_momentum", "h2h_home_winrate_10",
    # Contexte
    "rest_advantage", "month", "day_of_week",
    # Corners & Final Third
    "home_cornerkicks_avg_5", "away_cornerkicks_avg_5",
    "home_finalthirdentries_avg_5", "away_finalthirdentries_avg_5",
    # Interactions
    "elo_x_form_5", "elo_x_goals_5",
]

print(f"""
╔══════════════════════════════════════════════════════════════╗
║           CONFIGURATION FEATURES XGBOOST CHARGÉE              ║
╚══════════════════════════════════════════════════════════════╝
 Nombre de features :
   • Version MINIMAL  : {len(XGBOOST_FEATURES_MINIMAL)} features
   • Version MEDIUM   : {len(XGBOOST_FEATURES_MEDIUM)} features  
   • Version V1 (NO_XG) : {len(XGBOOST_FEATURES_NO_XG_V1)} features
   • Version V1 (WITH_XG) : {len(XGBOOST_FEATURES_WITH_XG_V1)} features
""")
