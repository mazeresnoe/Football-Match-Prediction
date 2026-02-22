⚽ Football Match Prediction System

End-to-end ML pipeline for predicting football match outcomes and identifying profitable betting opportunities
---

## 🎯 Project Overview
This project implements an end-to-end Machine Learning pipeline to predict football match outcomes (Home Win / Draw / Away Win) with the ultimate goal of beating bookmakers on specific matches where the model has an edge.
The Challenge
Can a machine learning model identify betting opportunities that outperform professional bookmakers?
The answer is nuanced: while the model cannot beat bookmakers on ALL matches (they have access to more data, insider information, and sophisticated models), it CAN identify SPECIFIC matches where it has an informational advantage.

## Key Objectives

- 🎲 Predicts match outcomes (Home/Draw/Away) with calibrated probabilities
- 💰 Identify value bets where model probability > bookmaker implied probability
- 📊 Optimize betting strategies to maximize ROI
---

## 📈 Results at a Glance

### Model Performance (TEST Set)

| Model | Log Loss ↓ | Brier Score ↓ | Accuracy | Overfitting/underfitting Gap |
|-------|-----------|---------------|----------|-----------------|
| **Bookmaker (Proportional)** | **1.0016** ⭐ | **0.5993** ⭐ | 49.95% | N/A |
| **Logistic Regression** | 1.0181 | 0.6097 | 49.15% | underfit |
| **XGBoost Ensemble** | 1.0210 | **0.6089**  | 48.98% | around 2% overfit  |
| XGBoost Baseline | 1.0332 | 0.6203 | 47.41% | Higher |
| Simple Elo | 1.1448 | 0.6589 | 48.90% | N/A |

**Key Insights**:
- ✅ **Bookmaker remains king** (insider info: lineups, data on players, injuries, market dynamics)
- ✅ **XGBoost Ensemble has best Brier Score** (0.6089) among ML models → better probability calibration
- ⚠️ **LogReg appears competitive** (1.0181 Log Loss) but **underfits** (model too simple)
- ✅ XGBoost will then be used

### Value Betting Strategies

**Test Set Analysis** (1.79 years, 4,628 matches with odds):

| Strategy | Bets/Year | Win Rate | ROI | Annual Profit (10€/bet) |
|----------|-----------|----------|-----|-------------------------|
| **Pure ROI** | 13 | 66.67% | **+16.62%** | +40€ |
| **Volume** | 111 | 51.01% | **+7.97%** | +158€ |

**Recommendation**: Volume strategy generates **4x more total profit** through diversification.

---

## 🛠️ Tech Stack

- **Python 3.11** | **Pandas 2.3** | **NumPy 2.3**
- **XGBoost 3.1** (Gradient Boosting) | **Optuna 4.6** (Hyperparameter Optimization)
- **Playwright 1.54** (Web Scraping) | **scikit-learn 1.7** (Baselines & Calibration)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mazeresnoe/Football-Match-Prediction.git
cd Football-Match-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Train model (includes feature selection, hyperparameter optimization, ensemble training)
python models/xgboost/core/step2b_optimization.py

# Optimize betting strategy
python models/xgboost/core/optimize_strategy.py

# Detect value bets
python models/xgboost/core/value_bet_extractor_v2.py
```

**Output**: CSV file with profitable betting opportunities

---

## 📂 Project Structure

```
Football-Match-Prediction/
├── .gitignore
├── requirements.txt
├── README.md                # This file
│
├── data/                    # Not in Git (too large)
│   ├── raw/                # 42,911 scraped matches (2015-2026)
│   │   ├── event_ids/      # Match IDs by league/season
│   │   ├── match_stats/    # Detailed statistics per match
│   │   └── merged/         # Combined dataset
│   ├── clean/              # Processed datasets
│   │   ├── post_match/     # After basic cleaning
│   │   └── prematch/       # With engineered features
│   │       ├── etape1/     # After feature engineering V1
│   │       ├── etape2/     # After feature engineering V2
│   │       │   └── split/  # no_xG (36,940) & xG (9,811)
│   │       └── etape3/     # Final clean (after EDA2)
│   ├── odds/               # Betting odds from football-data.co.uk
│   └── eda/                # Exploratory data analysis outputs
│
├── src/
│   ├── scraping/           # Data collection
│   │   ├── sofascore_scrap/  # Playwright automation
│   │   │   ├── round_scrapper_2.py      # Event ID scraper
│   │   │   └── match_stats_scraper.py   # Stats scraper
│   │   └── odds/           # Odds scraping + team mapping
│   ├── feature_engineering/  # 270+ features
│   │   └── etape1/         # Feature builders (V1 & V2)
│   └── utils/
│       └── mapping/        # Leagues ids, season ids, max_match ids
│
├── preprocessing/          # Data cleaning pipeline (8 steps)
│   ├── 1_etape/           # Raw merging & basic cleaning
│   │   ├── 1_raw_merger.py
│   │   ├── 2_cleaning.py
│   │   └── eda/           # Initial EDA
│   └── etape2/            # Feature engineering, EDA, cleaning
│       ├── 5_dual_dataset_splitter_v2.py
│       ├── 6_eda/         # Comprehensive EDA
│       ├── 7_cleaning.py
│       └── 8_feature_recovery.py
│
├── models/
│   ├── baseline/          # Elo, LogReg, Bookmaker baselines
│   ├── xgboost/          # Main ML pipeline
│   │   ├── core/         # Training & optimization scripts
│   │   ├── configs/      # Feature definitions
│   │   └── utils/        # Calibration, evaluation tools
│   ├── configs/          # Global configurations
│   ├── utils/            # Evaluation utilities
│   └── saved/            # Trained models (not in Git)
│
├── results/              # Model outputs (images in Git)
│   └── modeling/
│       ├── no_xg/        # Results without xG features
│       └── xg/           # Results with xG features
│
└── README/               # Detailed documentation
    ├── project_journey.md
    ├── architecture.md
    ├── lessons_learned.md
    ├── limitations.md
    └── future_work.md
```

---

## 📚 Documentation

Detailed documentation available in [`README/`](README/):

- 📖 [**Project Journey**](README/project_journey.md) - Complete development story with challenges & solutions
- 🏗️ [**Architecture**](README/architecture.md) - System design & complete workflow
- 🎓 [**Lessons Learned**](README/lessons_learned.md) - What worked, what didn't, key insights
- ⚠️ [**Limitations**](README/limitations.md) - Current constraints & known issues
- 🔮 [**Future Work**](README/future_work.md) - Planned improvements & roadmap

---

## 🔑 Key Features

### Data Pipeline
- ✅ **42,911 matches** scraped from SofaScore (Playwright automation with API interception)
- ✅ **36,940 clean matches** (2017-2026, no_xG dataset with enhanced stats)
- ✅ **9,811 xG matches** (2022-2026, xG stats introduced by SofaScore)
- ✅ **270+ engineered features** (Elo ratings, form metrics, rolling averages)
- ✅ **500+ team name variations** mapped to canonical names

### Model Pipeline
- ✅ **Feature Selection** (Minimal/Medium/Full - tested 3 configurations)
- ✅ **Hyperparameter Optimization** (Optuna, 100 trials, TimeSeriesSplit)
- ✅ **Ensemble Multi-Seed** (5 models with different seeds)
- ✅ **Isotonic Calibration** (trained on Train+CV combined)
- ✅ **Overfitting reduced ** (from around 10% to 2.13% accuracy gap)
  


### Betting Strategy
- ✅ **Automated value bet detection** (532 combinations tested)
- ✅ **Dual strategies** (Pure ROI vs Volume)
- ✅ **Positive expected value** on selective matches

---

## 📊 Competitions Covered

### Top 5 European Leagues + Second Divisions
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** + Championship
- 🇪🇸 **La Liga** + La Liga 2
- 🇮🇹 **Serie A** + Serie B
- 🇩🇪 **Bundesliga** + 2. Bundesliga
- 🇫🇷 **Ligue 1** + Ligue 2

### European Competitions
- 🏆 **UEFA Champions League**
- 🥈 **UEFA Europa League**
- 🥉 **UEFA Conference League**
- 🌍 **FIFA Club World Cup**

**Total**: 14 competitions, 11 years of data (2015-2026)

---

## 💡 Why Can't We Beat Bookmakers Consistently?

**The Fundamental Challenge**:

Bookmakers have:
- 📊 **More data**: Lineups, injuries, suspensions, insider info
- 💰 **Market dynamics**: Odds adjust based on betting volume
- 🧠 **Decades of optimization**: Have a full team, and Proprietary models refined over years

**Our Edge**:
- 🎯 **Selective betting**: Find specific matches where model > bookmaker
- 📈 **Probability calibration**: Better estimated probabilities than simple odds conversion
- 🤖 **Automated detection**: Systematic value bet identification

**Result**: Can't beat ALL matches, but can profit on **selective opportunities** (+16.62% ROI).

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

- ❌ Betting is an addiction and can involves significant financial risk,
- ❌ No guarantees of profit
- ❌ Not financial advice, don't use it if your goal is to make money

**Data Sources**: 
- Match statistics: [SofaScore](https://www.sofascore.com)
- Betting odds: [football-data.co.uk](https://www.football-data.co.uk)
- In a future: Betting odds: [Oddsportal](https://www.oddsportal.com/football/)

---

## 📧 Contact

**GitHub**: [mazeresnoe](https://github.com/mazeresnoe)  
**Email**: your.email@example.com  

---

**Built with ❤️ for football and data science**  
*Last Updated*: February 2026
