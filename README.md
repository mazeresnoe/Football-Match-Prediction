# Football Match Prediction & Value Betting

XGBoost-based pipeline to predict football match outcomes and detect value bets across major European leagues.

---

## Overview

This project uses historical match stats (Sofascore), historical odds (football-data.co.uk), and live odds (The Odds API) to:

1. Build a calibrated XGBoost ensemble model (137 features, no xG)
2. Predict home/draw/away probabilities for upcoming matches
3. Detect value bets using two strategies based on Expected Value

---

## Pipelines

### `run_pipeline_historic.py` — once per week after a matchday
Scrapes, clean and feature engineering of historic data stats and odds, rebuilds the dataset.


### `run_model.py` — every 2 weeks
Baselines → XGBoost optimization (Optuna) → ensemble calibration → strategy grid search.

```bash
python run_model.py
python run_model.py --skip-baselines
python run_model.py --skip-optuna
```

### `run_prediction.py` — 2–3× per week
Fetches upcoming odds → normalizes team names → builds prediction dataset → outputs value bets.

```bash
python run_prediction.py
```

**Outputs:**
- `results/predictions/predictions_full.csv` — probabilities for all upcoming matches
- `results/predictions/value_bets_volume.csv` — volume strategy (EV > 12%, conf > 50%)
- `results/predictions/value_bets_pure.csv` — pure strategy (EV > 14%, conf > 65%)

---

## Model

- **Algorithm:** XGBoost ensemble (5 seeds) + Isotonic calibration
- **Features:** 137 — Elo, form, momentum, shots, H2H, corners, passes, rest days...
- **Target:** Home win / Draw / Away win
- **Split:** 60% train / 20% CV / 20% test (temporal)
- **Overfitting:** accuracy gap ~2%, Brier gap ~2.6%

---

## Betting strategies

| Strategy | EV min | Confidence min | Expected ROI | Volume |
|----------|--------|----------------|-------------|--------|
| Volume   | 12%    | 50%            | +8%         | ~110 bets/year |
| Pure     | 14%    | 65%            | +16%        | ~13 bets/year |

Both strategies target Away bets only.

---

## Data sources

| Source | Usage |
|--------|-------|
| [Sofascore](https://www.sofascore.com) | Match stats (scraped via Playwright) |
| [football-data.co.uk](https://www.football-data.co.uk) | Historical odds |
| [The Odds API](https://the-odds-api.com) | Future odds (Unibet, Betclic, Winamax) |

---

## Leagues covered

Premier League · Championship · Ligue 1 · Ligue 2 · La Liga · La Liga 2 · Serie A · Serie B · Bundesliga · 2. Bundesliga · UEFA competitions (champions league, europa league, conference league, and fifa club world cup)

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Notes

- The model does not use expected goals (xG) — only actual stats
- `team_mapping.py` handles name normalization across the 3 data sources (Sofascore is the reference)
- Models are auto-promoted to production after each `run_model.py`
