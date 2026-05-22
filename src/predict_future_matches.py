"""
Prédiction sur les matchs futurs + Value Bet Detection

Workflow :
1. Charge futur_match_odds.csv (matchs futurs avec cotes)
2. Récupère les features les plus récentes de chaque équipe depuis le dataset historique
3. Calcule les features dérivées (momentum, ratios, interactions Elo)
4. Lance le modèle entraîné
5. Applique la stratégie value bet optimisée
6. Sauvegarde les recommandations dans results/predictions/

Usage :
    python src/predict_future_matches.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import models.configs.global_config as cfg
from models.configs.save_paths import SavePaths
from models.xgboost.core.step2b_optimization import XGBoostImproved

# ============================================================
# CONFIG
# ============================================================

FUTUR_MATCH_ODDS_PATH = Path("data/futur_match_odds/futur_match_odds.csv")
HISTORICAL_DATA_PATH  = cfg.DATA_NO_XG
OUTPUT_DIR            = Path("results/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes Elo et features dérivées à recalculer
ELO_COLS = ['elo_home', 'elo_away', 'elo_diff', 'elo_diff_squared', 'elo_diff_abs', 'elo_diff_log']


# ============================================================
# ÉTAPE 1 — Récupérer les dernières features connues par équipe
# ============================================================

def get_latest_team_features(hist_df: pd.DataFrame) -> dict:
    """
    Pour chaque équipe, récupère la dernière ligne du dataset historique
    où elle apparaît (home ou away) et extrait toutes ses features.
    
    Retourne un dict : {team_name: {feature: value, ...}}
    """
    hist_df = hist_df.sort_values('date')
    team_features = {}

    # Features home (préfixe home_)
    home_feat_cols = [c for c in hist_df.columns if c.startswith('home_')]
    # Features away (préfixe away_)
    away_feat_cols = [c for c in hist_df.columns if c.startswith('away_')]

    # Toutes les équipes uniques
    all_teams = set(hist_df['home_team'].str.lower().str.strip().unique()) | set(hist_df['away_team'].str.lower().str.strip().unique())

    for team in all_teams:
        # Dernier match en tant que home
        home_rows = hist_df[hist_df['home_team'].str.lower().str.strip() == team]
        # Dernier match en tant que away
        away_rows = hist_df[hist_df['away_team'].str.lower().str.strip() == team]

        features = {}

        if not home_rows.empty:
            last_home = home_rows.iloc[-1]
            for col in home_feat_cols:
                feat_name = col  # garder le nom tel quel (home_form_5, etc.)
                features[feat_name] = last_home[col]
            # Elo depuis le dernier match home
            features['elo_home_last'] = last_home.get('elo_home', cfg.XGBOOST_PARAMS.get('random_state', 1500))
            features['last_match_date_home'] = last_home['date']

        if not away_rows.empty:
            last_away = away_rows.iloc[-1]
            for col in away_feat_cols:
                feat_name = col
                features[feat_name] = last_away[col]
            features['elo_away_last'] = last_away.get('elo_away', 1500)
            features['last_match_date_away'] = last_away['date']

        # Elo final = le plus récent entre home et away
        last_home_date = features.get('last_match_date_home', pd.Timestamp.min)
        last_away_date = features.get('last_match_date_away', pd.Timestamp.min)

        if last_home_date >= last_away_date:
            features['elo_current'] = features.get('elo_home_last', 1500)
        else:
            features['elo_current'] = features.get('elo_away_last', 1500)

        team_features[team] = features

    print(f"✓ Features récupérées pour {len(team_features)} équipes")
    return team_features


# ============================================================
# ÉTAPE 2 — Construire le dataset de prédiction
# ============================================================

def get_h2h_features(home: str, away: str, hist_df: pd.DataFrame, n: int = 10) -> dict:
    """Calcule les features H2H entre deux équipes depuis l'historique."""
    mask = (
        ((hist_df['home_team'] == home) & (hist_df['away_team'] == away)) |
        ((hist_df['home_team'] == away) & (hist_df['away_team'] == home))
    )
    h2h = hist_df[mask].sort_values('date').tail(n)

    if h2h.empty:
        return {
            'h2h_home_wins_10': 0, 'h2h_draws_10': 0, 'h2h_away_wins_10': 0,
            'h2h_goals_for_10': np.nan, 'h2h_goals_against_10': np.nan,
            'h2h_home_winrate_10': np.nan, 'h2h_momentum': np.nan, 'h2h_goal_diff_10': np.nan,
        }

    home_wins = draws = away_wins = 0
    goals_for = goals_against = 0

    for _, row in h2h.iterrows():
        if row['home_team'] == home:
            gf, ga = row.get('homescore', 0), row.get('awayscore', 0)
        else:
            gf, ga = row.get('awayscore', 0), row.get('homescore', 0)
        goals_for += gf or 0
        goals_against += ga or 0
        if gf > ga: home_wins += 1
        elif gf == ga: draws += 1
        else: away_wins += 1

    n_matches = len(h2h)
    winrate = home_wins / n_matches if n_matches > 0 else np.nan
    goal_diff = (goals_for - goals_against) / n_matches if n_matches > 0 else np.nan

    # H2H momentum : 5 derniers vs 10 derniers
    h2h_5 = h2h.tail(5)
    hw5 = sum(1 for _, r in h2h_5.iterrows() if
              (r['home_team'] == home and (r.get('homescore', 0) or 0) > (r.get('awayscore', 0) or 0)) or
              (r['away_team'] == home and (r.get('awayscore', 0) or 0) > (r.get('homescore', 0) or 0)))
    wr5 = hw5 / len(h2h_5) if len(h2h_5) > 0 else np.nan
    momentum = (wr5 - winrate) if pd.notna(wr5) and pd.notna(winrate) else np.nan

    return {
        'h2h_home_wins_10':    home_wins,
        'h2h_draws_10':        draws,
        'h2h_away_wins_10':    away_wins,
        'h2h_goals_for_10':    goals_for / n_matches if n_matches > 0 else np.nan,
        'h2h_goals_against_10': goals_against / n_matches if n_matches > 0 else np.nan,
        'h2h_home_winrate_10': winrate,
        'h2h_momentum':        momentum,
        'h2h_goal_diff_10':    goal_diff,
    }


def build_prediction_dataset(future_df: pd.DataFrame, team_features: dict,
                              hist_df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Pour chaque match futur, assemble les features du modèle
    en combinant les features home de l'équipe domicile
    et les features away de l'équipe visiteuse.
    """
    rows = []

    # Récupérer les Elo depuis le dataset historique directement
    # (les plus récentes par équipe)
    latest_elo = {}
    for _, row in hist_df.sort_values('date').iterrows():
        latest_elo[row['home_team']] = row.get('elo_home', 1500)
        latest_elo[row['away_team']] = row.get('elo_away', 1500)

    for _, match in future_df.iterrows():
        home = match['home_team'].lower().strip()
        away = match['away_team'].lower().strip()

        home_feats = team_features.get(home, {})
        away_feats = team_features.get(away, {})

        if not home_feats and not away_feats:
            continue  # Équipe inconnue, skip

        row = {
            'event_id':      match.get('event_id'),
            'date':          match.get('date'),
            'league':        match.get('league'),
            'season':        match.get('season'),
            'home_team':     home,
            'away_team':     away,
            'round':         match.get('round'),
            # Cotes
            'unibet_home':   match.get('unibet_home'),
            'unibet_draw':   match.get('unibet_draw'),
            'unibet_away':   match.get('unibet_away'),
            'betclic_home':  match.get('betclic_home'),
            'betclic_draw':  match.get('betclic_draw'),
            'betclic_away':  match.get('betclic_away'),
            'winamax_home':  match.get('winamax_home'),
            'winamax_draw':  match.get('winamax_draw'),
            'winamax_away':  match.get('winamax_away'),
        }

        # Features home (depuis le profil home de l'équipe domicile)
        for col in [c for c in model_features if c.startswith('home_')]:
            row[col] = home_feats.get(col, np.nan)

        # Features away (depuis le profil away de l'équipe visiteuse)
        for col in [c for c in model_features if c.startswith('away_')]:
            row[col] = away_feats.get(col, np.nan)

        # Elo
        elo_h = latest_elo.get(home, 1500)
        elo_a = latest_elo.get(away, 1500)
        row['elo_home']         = elo_h
        row['elo_away']         = elo_a
        row['elo_diff']         = elo_h - elo_a
        row['elo_diff_squared'] = (elo_h - elo_a) ** 2
        row['elo_diff_abs']     = abs(elo_h - elo_a)
        row['elo_diff_log']     = np.sign(elo_h - elo_a) * np.log1p(abs(elo_h - elo_a))

        # Features diff/dérivées (recalcul depuis home/away)
        def get(k): return row.get(k, np.nan)

        row['diff_form_5']            = get('home_form_5') - get('away_form_5') if pd.notna(get('home_form_5')) and pd.notna(get('away_form_5')) else np.nan
        row['diff_form_volatility']   = get('home_form_volatility') - get('away_form_volatility') if pd.notna(get('home_form_volatility')) and pd.notna(get('away_form_volatility')) else np.nan
        row['diff_momentum_form']     = get('home_momentum_form') - get('away_momentum_form') if pd.notna(get('home_momentum_form')) and pd.notna(get('away_momentum_form')) else np.nan
        row['diff_momentum_goals']    = get('home_momentum_goals') - get('away_momentum_goals') if pd.notna(get('home_momentum_goals')) and pd.notna(get('away_momentum_goals')) else np.nan
        row['diff_momentum_defense']  = get('home_momentum_defense') - get('away_momentum_defense') if pd.notna(get('home_momentum_defense')) and pd.notna(get('away_momentum_defense')) else np.nan
        row['diff_goals_conceded_avg_5']  = get('home_goals_conceded_avg_5') - get('away_goals_conceded_avg_5') if pd.notna(get('home_goals_conceded_avg_5')) and pd.notna(get('away_goals_conceded_avg_5')) else np.nan
        row['diff_goals_conceded_avg_10'] = get('home_goals_conceded_avg_10') - get('away_goals_conceded_avg_10') if pd.notna(get('home_goals_conceded_avg_10')) and pd.notna(get('away_goals_conceded_avg_10')) else np.nan
        row['diff_goal_diff_10']      = get('home_goal_diff_10') - get('away_goal_diff_10') if pd.notna(get('home_goal_diff_10')) and pd.notna(get('away_goal_diff_10')) else np.nan
        row['diff_shot_conversion_5'] = get('home_shot_conversion_5') - get('away_shot_conversion_5') if pd.notna(get('home_shot_conversion_5')) and pd.notna(get('away_shot_conversion_5')) else np.nan
        row['diff_bigchance_created_10'] = get('home_bigchancecreated_avg_10') - get('away_bigchancecreated_avg_10') if pd.notna(get('home_bigchancecreated_avg_10')) and pd.notna(get('away_bigchancecreated_avg_10')) else np.nan
        row['diff_bigchance_conversion'] = get('home_bigchance_conversion_10') - get('away_bigchance_conversion_10') if pd.notna(get('home_bigchance_conversion_10')) and pd.notna(get('away_bigchance_conversion_10')) else np.nan
        row['diff_corners_5']         = get('home_cornerkicks_avg_5') - get('away_cornerkicks_avg_5') if pd.notna(get('home_cornerkicks_avg_5')) and pd.notna(get('away_cornerkicks_avg_5')) else np.nan
        row['diff_corners_10']        = get('home_cornerkicks_avg_10') - get('away_cornerkicks_avg_10') if pd.notna(get('home_cornerkicks_avg_10')) and pd.notna(get('away_cornerkicks_avg_10')) else np.nan
        row['diff_finalthird_10']     = get('home_finalthirdentries_avg_10') - get('away_finalthirdentries_avg_10') if pd.notna(get('home_finalthirdentries_avg_10')) and pd.notna(get('away_finalthirdentries_avg_10')) else np.nan
        row['diff_defense_efficiency']= get('home_defense_efficiency_10') - get('away_defense_efficiency_10') if pd.notna(get('home_defense_efficiency_10')) and pd.notna(get('away_defense_efficiency_10')) else np.nan
        row['diff_attack_defense_ratio'] = get('home_attack_defense_ratio') - get('away_attack_defense_ratio') if pd.notna(get('home_attack_defense_ratio')) and pd.notna(get('away_attack_defense_ratio')) else np.nan

        # Interactions Elo
        diff = row['elo_diff']
        row['elo_x_form_5']    = diff * row.get('diff_form_5', np.nan) if pd.notna(row.get('diff_form_5')) else np.nan
        row['elo_x_form_10']   = diff * (get('home_form_10') - get('away_form_10')) if pd.notna(get('home_form_10')) and pd.notna(get('away_form_10')) else np.nan
        row['elo_x_goals_5']   = diff * (get('home_goals_scored_avg_5') - get('away_goals_scored_avg_5')) if pd.notna(get('home_goals_scored_avg_5')) and pd.notna(get('away_goals_scored_avg_5')) else np.nan
        row['elo_x_goals_10']  = diff * (get('home_goals_scored_avg_10') - get('away_goals_scored_avg_10')) if pd.notna(get('home_goals_scored_avg_10')) and pd.notna(get('away_goals_scored_avg_10')) else np.nan

        # Temporal features
        match_date = pd.to_datetime(match['date'])
        row['month']       = match_date.month
        row['day_of_week'] = match_date.dayofweek
        row['season_month'] = (match_date.month - 8) % 12  # saison commence en août

        # rest_advantage : pas de donnée live → on met 7 par défaut
        row['days_rest_home'] = 7
        row['days_rest_away'] = 7
        row['rest_advantage'] = 0
        row['elo_x_rest'] = 0

        # Log transforms
        for base, log_col in [
            ('rest_advantage', 'rest_advantage_log'),
            ('diff_goal_diff_10', 'diff_goal_diff_10_log'),
            ('elo_x_form_5', 'elo_x_form_5_log'),
            ('elo_x_form_10', 'elo_x_form_10_log'),
            ('elo_x_goals_5', 'elo_x_goals_5_log'),
            ('elo_x_goals_10', 'elo_x_goals_10_log'),
            ('elo_x_rest', 'elo_x_rest_log'),
            ('elo_diff_squared', 'elo_diff_squared_log'),
            ('home_attack_defense_ratio', 'home_attack_defense_ratio_log'),
            ('away_attack_defense_ratio', 'away_attack_defense_ratio_log'),
            ('diff_attack_defense_ratio', 'diff_attack_defense_ratio_log'),
        ]:
            val = row.get(base, np.nan)
            row[log_col] = np.sign(val) * np.log1p(abs(val)) if pd.notna(val) else np.nan

        # H2H features
        h2h = get_h2h_features(home, away, hist_df, n=10)
        row.update(h2h)

        rows.append(row)

    df_pred = pd.DataFrame(rows)
    print(f"✓ {len(df_pred)} matchs assemblés pour prédiction")
    return df_pred


# ============================================================
# ÉTAPE 3 — Prédiction + Value Bet Detection
# ============================================================

def predict_and_detect_value_bets(pred_df: pd.DataFrame, model, model_features: list,
                                   strategy: dict) -> pd.DataFrame:
    """
    Lance le modèle et applique la stratégie value bet.
    Utilise la moyenne des 3 bookmakers pour les cotes.
    """
    min_ev         = strategy['min_ev']
    min_confidence = strategy['min_confidence']
    bet_types      = strategy.get('bet_types', None)

    # Prépare X avec les features du modèle (NaN → 0 pour les features manquantes)
    X = pred_df[model_features].copy()
    missing_pct = X.isna().mean().mean()
    if missing_pct > 0:
        print(f"  ⚠ {missing_pct*100:.1f}% de valeurs manquantes → imputées à 0")
        X = X.fillna(0)

    probs = model.predict_proba(X)  # shape (n, 3) → [Home, Draw, Away]

    value_bets = []

    for idx, (_, row) in enumerate(pred_df.iterrows()):
        model_prob = probs[idx]  # [p_home, p_draw, p_away]

        # Cote = moyenne des 3 bookmakers (ou meilleure dispo)
        def best_odd(h, b, w):
            vals = [v for v in [h, b, w] if pd.notna(v) and v > 1]
            return max(vals) if vals else np.nan

        odds = [
            best_odd(row.get('unibet_home'), row.get('betclic_home'), row.get('winamax_home')),
            best_odd(row.get('unibet_draw'), row.get('betclic_draw'), row.get('winamax_draw')),
            best_odd(row.get('unibet_away'), row.get('betclic_away'), row.get('winamax_away')),
        ]

        outcomes = [('Home', 0), ('Draw', 1), ('Away', 2)]

        for bet_type, i in outcomes:
            if bet_types and bet_type not in bet_types:
                continue

            odd = odds[i]
            prob = model_prob[i]

            if not pd.notna(odd) or odd <= 1:
                continue

            ev = (prob * odd) - 1

            if ev > min_ev and prob > min_confidence:
                # Meilleur bookmaker pour cette cote
                if i == 0:
                    bookie_odds = {'Unibet': row.get('unibet_home'), 'Betclic': row.get('betclic_home'), 'Winamax': row.get('winamax_home')}
                elif i == 1:
                    bookie_odds = {'Unibet': row.get('unibet_draw'), 'Betclic': row.get('betclic_draw'), 'Winamax': row.get('winamax_draw')}
                else:
                    bookie_odds = {'Unibet': row.get('unibet_away'), 'Betclic': row.get('betclic_away'), 'Winamax': row.get('winamax_away')}

                best_bookie = max(
                    {k: v for k, v in bookie_odds.items() if pd.notna(v) and v > 1},
                    key=lambda k: bookie_odds[k],
                    default='N/A'
                )

                value_bets.append({
                    'event_id':      row.get('event_id'),
                    'date':          row.get('date'),
                    'league':        row.get('league'),
                    'home_team':     row.get('home_team'),
                    'away_team':     row.get('away_team'),
                    'round':         row.get('round'),
                    'bet_type':      bet_type,
                    'model_prob':    round(prob, 4),
                    'best_odd':      round(odd, 2),
                    'best_bookmaker':best_bookie,
                    'ev_pct':        round(ev * 100, 2),
                    'edge':          round(prob - (1 / odd), 4),
                    'implied_prob':  round(1 / odd, 4),
                    # Cotes par bookmaker
                    'unibet':        bookie_odds.get('Unibet'),
                    'betclic':       bookie_odds.get('Betclic'),
                    'winamax':       bookie_odds.get('Winamax'),
                    # Probas du modèle
                    'prob_home':     round(model_prob[0], 4),
                    'prob_draw':     round(model_prob[1], 4),
                    'prob_away':     round(model_prob[2], 4),
                })

    if not value_bets:
        return pd.DataFrame()
    return pd.DataFrame(value_bets).sort_values('ev_pct', ascending=False).reset_index(drop=True)


def filter_valid_matches(pred_df: pd.DataFrame, model_features: list, max_missing_pct: float = 0.30) -> pd.DataFrame:
    """Exclut les matchs où plus de 30% des features sont à zéro/NaN."""
    feature_df = pred_df[model_features]
    missing_pct = (feature_df.isna() | (feature_df == 0)).mean(axis=1)
    valid = missing_pct <= max_missing_pct
    n_excluded = (~valid).sum()
    if n_excluded > 0:
        print(f"  ⚠ {n_excluded} matchs exclus (features insuffisantes) :")
        for _, row in pred_df[~valid].iterrows():
            print(f"    - {row['home_team']} vs {row['away_team']} ({row['league']})")
    return pred_df[valid].reset_index(drop=True)

# ============================================================
# MAIN
# ============================================================

def load_strategy(filename: str, default: dict) -> dict:
    path = SavePaths.get_result_path('production/strategy', filename)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

DEFAULT_STRATEGY = {'min_ev': 0.08, 'min_confidence': 0.50, 'bet_types': None,
                    'expected_roi': 0, 'expected_win_rate': 0, 'name': 'Default'}

def main():
    print("=" * 65)
    print("  PRÉDICTIONS MATCHS FUTURS + VALUE BET DETECTION")
    print("=" * 65)

    print("\n[1/5] Chargement du modèle...")
    model_path = SavePaths.get_latest_model('production') or SavePaths.get_latest_model('experiments')
    if model_path is None:
        print("❌ Aucun modèle trouvé.")
        return
    model_data     = joblib.load(model_path)
    model          = model_data['model']
    model_features = model_data['features']
    print(f"   Modèle : {model_path.name} ({len(model_features)} features)")

    print("\n[2/5] Chargement des stratégies...")
    strategy_volume = load_strategy('best_strategy.json', DEFAULT_STRATEGY)
    strategy_pure   = load_strategy('best_strategy_pure.json', DEFAULT_STRATEGY)
    for s in [strategy_volume, strategy_pure]:
        print(f"   [{s.get('name','?')}] EV>{s['min_ev']*100:.1f}% | Conf>{s['min_confidence']*100:.1f}% | ROI attendu {s.get('expected_roi',0):+.1f}%")

    print("\n[3/5] Chargement des matchs futurs avec cotes...")
    if not FUTUR_MATCH_ODDS_PATH.exists():
        print(f"❌ Fichier introuvable : {FUTUR_MATCH_ODDS_PATH}")
        return
    future_df = pd.read_csv(FUTUR_MATCH_ODDS_PATH)
    print(f"   {len(future_df)} matchs chargés")

    print("\n[4/5] Extraction des features depuis l'historique...")
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
    team_features = get_latest_team_features(hist_df)

    pred_df = build_prediction_dataset(future_df, team_features, hist_df, model_features)
    if pred_df.empty:
        print("❌ Impossible de construire le dataset de prédiction.")
        return

    # Filtre les matchs avec features insuffisantes
    pred_df = filter_valid_matches(pred_df, model_features)
    if pred_df.empty:
        print("❌ Aucun match avec features suffisantes.")
        return

    print("\n[5/5] Prédiction et détection des value bets...")
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Prédictions complètes
    probs_temp = model.predict_proba(pred_df[model_features].fillna(0))
    pred_df['prob_home'] = probs_temp[:, 0].round(4)
    pred_df['prob_draw'] = probs_temp[:, 1].round(4)
    pred_df['prob_away'] = probs_temp[:, 2].round(4)
    pred_df['predicted_outcome'] = pd.Series(probs_temp.argmax(axis=1)).map({0: 'Home', 1: 'Draw', 2: 'Away'}).values

    pred_cols = ['event_id', 'date', 'league', 'home_team', 'away_team', 'round',
                 'unibet_home', 'unibet_draw', 'unibet_away',
                 'prob_home', 'prob_draw', 'prob_away', 'predicted_outcome']
    pred_df[pred_cols].to_csv(OUTPUT_DIR / "predictions_full.csv", index=False)
    print(f"✅ Prédictions complètes : results/predictions/predictions_full.csv")

    # Stratégie Volume
    vb_volume = predict_and_detect_value_bets(pred_df, model, model_features, strategy_volume)
    vb_volume.to_csv(OUTPUT_DIR / "value_bets_volume.csv", index=False)
    print(f"\n[Stratégie Volume] {len(vb_volume)} value bets → value_bets_volume.csv")
    if not vb_volume.empty:
        print(vb_volume[['date','league','home_team','away_team','bet_type','model_prob','best_odd','ev_pct']].head(10).to_string(index=False))

    # Stratégie Pure
    vb_pure = predict_and_detect_value_bets(pred_df, model, model_features, strategy_pure)
    vb_pure.to_csv(OUTPUT_DIR / f"value_bets_pure.csv", index=False)
    print(f"\n[Stratégie Pure] {len(vb_pure)} value bets → value_bets_pure_{date_str}.csv")
    if not vb_pure.empty:
        print(vb_pure[['date','league','home_team','away_team','bet_type','model_prob','best_odd','ev_pct']].head(10).to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  TERMINÉ — {len(pred_df)} matchs analysés")
    print(f"  Volume : {len(vb_volume)} paris | Pure : {len(vb_pure)} paris")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
