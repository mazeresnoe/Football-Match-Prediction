"""
Diagnostic : identifie les matchs avec features manquantes ou valeurs suspectes.
Lance : python src/debug_features.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import models.configs.global_config as cfg

FUTUR_MATCH_ODDS_PATH = Path("data/futur_match_odds/futur_match_odds.csv")

print("=" * 65)
print("  DIAGNOSTIC FEATURES")
print("=" * 65)

# ── 1. Charger les données ──────────────────────────────────
hist_df = pd.read_csv(cfg.DATA_NO_XG)
hist_df['date'] = pd.to_datetime(hist_df['date'])
hist_df = hist_df.sort_values('date')

future_df = pd.read_csv(FUTUR_MATCH_ODDS_PATH)

print(f"\nHistorique : {len(hist_df)} matchs")
print(f"Futurs     : {len(future_df)} matchs")

# ── 2. Vérifier noms dans historique ───────────────────────
hist_teams = set(hist_df['home_team'].str.lower().str.strip()) | \
             set(hist_df['away_team'].str.lower().str.strip())

print(f"\nÉquipes uniques dans historique : {len(hist_teams)}")

# ── 3. Vérifier noms dans futurs ───────────────────────────
print("\n[CHECK 1] Équipes dans futur_match_odds absentes de l'historique :")
missing = []
for _, row in future_df.iterrows():
    h = row['home_team'].lower().strip()
    a = row['away_team'].lower().strip()
    if h not in hist_teams:
        missing.append(('HOME', h, row['away_team'], row['league']))
    if a not in hist_teams:
        missing.append(('AWAY', row['home_team'], a, row['league']))

if missing:
    for side, h, a, league in missing:
        print(f"  {side} MISSING: {repr(h if side=='HOME' else a)} ({league})")
else:
    print("  ✓ Tous les noms matchent")

# ── 4. Construire team_features et vérifier ────────────────
print("\n[CHECK 2] Construction du dict team_features...")

home_feat_cols = [c for c in hist_df.columns if c.startswith('home_') and c != 'home_team']
away_feat_cols = [c for c in hist_df.columns if c.startswith('away_') and c != 'away_team']

team_features = {}
all_teams = set(hist_df['home_team'].str.lower().str.strip()) | \
            set(hist_df['away_team'].str.lower().str.strip())

for team in all_teams:
    home_rows = hist_df[hist_df['home_team'].str.lower().str.strip() == team]
    away_rows = hist_df[hist_df['away_team'].str.lower().str.strip() == team]
    features = {}
    if not home_rows.empty:
        last = home_rows.iloc[-1]
        for col in home_feat_cols:
            features[col] = last[col]
        features['elo_home_last'] = last.get('elo_home', 1500)
        features['last_match_date_home'] = last['date']
    if not away_rows.empty:
        last = away_rows.iloc[-1]
        for col in away_feat_cols:
            features[col] = last[col]
        features['elo_away_last'] = last.get('elo_away', 1500)
        features['last_match_date_away'] = last['date']
    team_features[team] = features

print(f"  ✓ {len(team_features)} équipes dans team_features")

# ── 5. Vérifier features pour chaque match futur ──────────
print("\n[CHECK 3] Features home_form_5 pour chaque match futur :")
print(f"{'Match':<50} {'home_form_5':>12} {'away_form_5':>12} {'status':>10}")
print("-" * 90)

suspicious = []
for _, match in future_df.iterrows():
    home = match['home_team'].lower().strip()
    away = match['away_team'].lower().strip()

    hf = team_features.get(home, {})
    af = team_features.get(away, {})

    home_form = hf.get('home_form_5', 'MISSING')
    away_form = af.get('away_form_5', 'MISSING')

    status = "OK"
    if home_form == 'MISSING' or away_form == 'MISSING':
        status = "⚠ MISSING"
        suspicious.append((home, away, match['league'], 'MISSING_FEATURE'))
    elif pd.isna(home_form) or pd.isna(away_form):
        status = "⚠ NaN"
        suspicious.append((home, away, match['league'], 'NaN_FEATURE'))

    match_str = f"{home[:22]} vs {away[:22]}"
    hf_str = f"{home_form:.4f}" if isinstance(home_form, float) and not pd.isna(home_form) else str(home_form)
    af_str = f"{away_form:.4f}" if isinstance(away_form, float) and not pd.isna(away_form) else str(away_form)
    print(f"  {match_str:<48} {hf_str:>12} {af_str:>12} {status:>10}")

# ── 6. Résumé ──────────────────────────────────────────────
print(f"\n[RÉSUMÉ]")
print(f"  Matchs totaux      : {len(future_df)}")
print(f"  Matchs suspects    : {len(suspicious)}")

if suspicious:
    print("\n  Matchs avec problème :")
    for h, a, league, reason in suspicious:
        print(f"    {reason}: {h} vs {a} ({league})")

# ── 7. Vérifier les Elo ───────────────────────────────────
print("\n[CHECK 4] Elo depuis l'historique :")
latest_elo = {}
for _, row in hist_df.iterrows():
    latest_elo[row['home_team'].lower().strip()] = row.get('elo_home', 1500)
    latest_elo[row['away_team'].lower().strip()] = row.get('elo_away', 1500)

elo_default = []
for _, match in future_df.iterrows():
    home = match['home_team'].lower().strip()
    away = match['away_team'].lower().strip()
    elo_h = latest_elo.get(home, None)
    elo_a = latest_elo.get(away, None)
    if elo_h is None or elo_a is None:
        elo_default.append((home, away))

if elo_default:
    print(f"  ⚠ {len(elo_default)} matchs avec Elo par défaut (1500) :")
    for h, a in elo_default:
        print(f"    {h} vs {a}")
else:
    print(f"  ✓ Elo trouvé pour toutes les équipes")

print("\n" + "=" * 65)
print("  FIN DIAGNOSTIC")
print("=" * 65)