# 🏗️ System Architecture

> **Technical design and workflow of the Football Match Prediction System**

---

## 📊 High-Level Overview

```
┌─────────────────┐
│  Data Sources   │
│  (SofaScore,    │
│   Odds Sites)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Scraping      │
│  (Playwright +  │
│   API Intercept)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Data       │
│  42,911 matches │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  8-Step         │
│  Preprocessing  │
└────────┬────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │ no_xG  │    │  xG    │    │  EDA   │
    │36,940  │    │ 9,811  │    │Reports │
    └───┬────┘    └───┬────┘    └────────┘
        │             │
        └──────┬──────┘
               ▼
    ┌──────────────────┐
    │ Feature          │
    │ Engineering      │
    │ (270+ features)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Model Training  │
    │  (XGBoost        │
    │   Ensemble)      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Value Betting    │
    │ Detection        │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Betting          │
    │ Opportunities    │
    └──────────────────┘
```

---

## 🔄 Complete Workflow

### **Step 1: Team Mapping Setup**

**Purpose**: Standardize team names across different data sources

**Input**: Raw team names from SofaScore and football-data.co.uk
**Output**: Canonical team names + mapping dictionary

**File**: `src/utils/mapping/map.py`

**Process**:
```python
# 500+ manual mappings
TEAM_NAME_MAPPING = {
    "Man United": "manchester united",
    "Man City": "manchester city",
    "Bayern Munich": "fc bayern münchen",
    "PSV": "psv eindhoven",
    "Ath Madrid": "atlético madrid",
    # ... 495+ more
}
```

**Challenges**:
- Different sources use different conventions
- Special characters (ü, é, ñ)
- Abbreviations vs full names
- Historical name changes

**Location**: Applied in:
- `1_raw_merger.py` (SofaScore data)
- `team_mapping_complete.py` (odds data)

---

### **Step 2: Data Scraping**

#### **2.1 Match Statistics (SofaScore)**

**Challenge**: JavaScript-heavy website with internal API calls

**Solution**: Playwright with API response interception

**Architecture**:
```
Playwright Browser
       ↓
Navigate to page
       ↓
Intercept network traffic
       ↓
Filter API responses
       ↓
Extract JSON data
       ↓
Save to file
```

**Implementation**:

**Scraper 1: Event IDs** (`round_scrapper_2.py`)
```python
def fetch_rounds_info(tournament_id, season_id):
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"
    
    with sync_playwright() as p:
        page = p.chromium.launch()
        
        def handle_response(response):
            if url in response.url and response.status == 200:
                data = response.json()
                # Extract round info
        
        page.on("response", handle_response)
        page.goto(url, wait_until="networkidle")
```

**Output**: JSON files per league/season with event IDs
- Location: `data/raw/event_ids/{league}/{season}.json`
- Structure:
```json
[
  {
    "event_id": 12345,
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "date": "2024-01-15T15:00:00",
    "round": 20,
    "homeScore": 2,
    "awayScore": 1,
    "description": "Ended"
  }
]
```

**Scraper 2: Match Statistics** (`match_stats_scraper.py`)
```python
def scrape_match_statistics(event_id):
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
    
    def handle_response(response):
        if url in response.url and response.status == 200:
            match_stats = response.json()
            # Extract ALL period stats
```

**Output**: CSV files per league/season with detailed stats
- Location: `data/raw/match_stats/{league}/{season}_stats.csv`
- Contains: 100+ statistical columns per match

**Progress Tracking**:
```python
# Save progress every 10 matches
progress = {
    "scraped_events": [12345, 12346, ...],
    "failed_events": [12399],
    "last_update": "2024-01-15T10:30:00"
}
# Location: data/raw/scraping_progress/{league}_{season}_progress.json
```

#### **2.2 Betting Odds (football-data.co.uk)**

**Challenge**: Simple HTML, easy scraping

**Solution**: requests + pandas

**File**: `scrape_odds.py`

```python
def scrape_season(league_code, season):
    # Format: https://www.football-data.co.uk/mmz4281/2122/E0.csv
    year_short = season.split('-')[0][-2:] + season.split('-')[1]
    url = f"{BASE_URL}/mmz4281/{year_short}/{league_code}.csv"
    
    response = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(response.text))
    return df
```

**Output**: CSV files in `data/odds/raw/{league}.csv`

**Cleaning** (`team_mapping_complete.py`):
```python
# Apply same team mapping as SofaScore
odds['home_team'] = odds['home_team'].map(TEAM_NAME_MAPPING)
odds['away_team'] = odds['away_team'].map(TEAM_NAME_MAPPING)

# Select best bookmaker (priority: Pinnacle > Bet365 > Average)
odds = odds[['date', 'home_team', 'away_team', 'PSH', 'PSD', 'PSA']]
odds.rename(columns={'PSH': 'odds_home', 'PSD': 'odds_draw', 'PSA': 'odds_away'})
```

**Output**: `data/odds/all_odds_standardized.csv`

---

### **Step 3: Data Merging**

**Purpose**: Combine all league CSVs into single dataset

**File**: `preprocessing/1_etape/1_raw_merger.py`

**Process**:
1. Load all match_stats CSVs from `data/raw/match_stats/`
2. Normalize column names (lowercase, underscores)
3. Apply team name mapping
4. Add metadata (league, season, source_file)
5. Combine all into single DataFrame
6. Sort by date

**Output**: `data/raw/merged/dataset_raw_merged.csv` (42,911 matches)

---

### **Step 4: Initial EDA**

**Purpose**: Understand raw data quality

**File**: `preprocessing/1_etape/eda/eda_observation.py`

**Analyses**:
- Missing value patterns per column
- Feature distributions (mean, std, min, max)
- Outliers (Z-score > 3)
- League/season patterns
- **Correlations sorted** (identify redundant features)

**Output**: `preprocessing/1_etape/eda/` (CSV reports + plots)

---

### **Step 5: Basic Cleaning**

**Purpose**: Remove invalid/duplicate data

**File**: `preprocessing/1_etape/2_cleaning.py`

**Steps**:
1. Drop 100% empty columns
2. Remove canceled/postponed matches (status filter)
3. Standardize date formats (YYYY-MM-DD)
4. Convert numeric columns automatically
5. Remove duplicates (same date + teams)
6. Normalize team names (lowercase, strip)
7. Reorder columns (ID columns first)

**Output**: `data/clean/post_match/post_match_clean.csv` (42,911 matches)

---

### **Step 6: Feature Engineering V1 (Basic)**

**Purpose**: Create foundational features

**File**: `src/feature_engineering/etape1/3_feature_eng_v1_basic.py`

**Features Created**:

**1. Elo Ratings**:
```python
def update_elo(winner_elo, loser_elo, K=20):
    expected = 1 / (1 + 10**((loser_elo - winner_elo)/400))
    winner_new = winner_elo + K * (1 - expected)
    loser_new = loser_elo + K * (0 - (1 - expected))
    return winner_new, loser_new

# Initial: 1500, Home advantage: +100
```

**2. Form Metrics**:
- Last 5/10 matches performance
- Points, goals scored/conceded
- Win rate, clean sheet rate
- Weighted by recency

**3. Rolling Averages**:
- Goals, shots, possession (per match)
- Averages over 5/10 match windows
- For ALL available stats (100+)

**4. Head-to-Head**:
- Last 5/10 encounters between teams
- Win/draw/loss counts
- Goal differentials in H2H

**Output**: `data/clean/prematch/etape1/full_dataset.csv`

**Key Design**:
- **Chronological processing** (no future leakage)
- **Min matches threshold** (5 matches minimum for features)
- **NaN for insufficient data** (model handles missing)

---

### **Step 7: Feature Engineering V2 (Advanced)**

**Purpose**: Create complex derived features

**File**: `src/feature_engineering/etape1/4_feature_eng_v2_advanced.py`

**Categories**:

**1. Temporal Features**:
```python
df['month'] = df['date'].dt.month
df['season_stage'] = pd.cut(df['season_month'], bins=[0, 3, 7, 100], labels=[0,1,2])
df['days_rest_home'] = df.groupby('home_team')['date'].diff().dt.days
df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
```

**2. Elo Interactions**:
```python
df['elo_x_form_5'] = df['elo_diff'] * df['diff_form_5']
df['elo_diff_squared'] = df['elo_diff'] ** 2
df['elo_diff_log'] = np.sign(df['elo_diff']) * np.log1p(abs(df['elo_diff']))
```

**3. Momentum** (Short-term vs Long-term):
```python
df['home_momentum_form'] = df['home_form_5'] - df['home_form_10']
df['home_momentum_goals'] = df['home_goals_avg_5'] - df['home_goals_avg_10']
```

**4. Efficiency Ratios**:
```python
df['home_shot_conversion'] = df['home_goals_avg_5'] / df['home_shots_avg_5']
df['home_defense_efficiency'] = df['home_goals_conceded_avg_10'] / df['home_shots_conceded_avg_10']
```

**5. Balance Features**:
```python
df['home_attack_defense_ratio'] = df['home_goals_avg_10'] / df['home_goals_conceded_avg_10']
```

**6. xG Features** (if available):
```python
# Always create columns, fill NaN for unavailable
df['home_xg_overperf_10'] = df['home_goals_avg_10'] - df['home_xg_avg_10']
df['xg_momentum'] = df['diff_xg_5'] - df['diff_xg_10']

# Data completeness scores
df['has_xg_stats'] = df[xg_cols].notna().mean(axis=1)
df['has_xg_stats_flag'] = (df['has_xg_stats'] > 0.1).astype(int)
```

**7. Completeness Scores**:
```python
df['stats_completeness_score'] = df[all_feature_cols].notna().mean(axis=1)
```

**Output**: `data/clean/prematch/etape2/full_dataset_v2.csv`

**Total Features**: ~270 (varies by data availability)

---

### **Step 8: Dataset Splitting**

**Purpose**: Create no_xG and xG datasets

**File**: `preprocessing/etape2/5_dual_dataset_splitter_v2.py`

**Logic**:
```python
# Detect xG columns automatically
xg_cols = [c for c in df.columns if 'xg' in c.lower() or 'expectedgoals' in c.lower()]

# NO_XG: All rows, drop xG columns
df_no_xg = df.copy()
df_no_xg.drop(columns=xg_cols, inplace=True)

# WITH_XG: Only rows with xG data (has_xg_stats >= 0.1)
df_with_xg = df[df['has_xg_stats'] >= 0.1].copy()
```

**Output**:
- `data/clean/prematch/etape2/split/full_dataset_no_xg.csv` (36,940 matches, 2017-2026)
- `data/clean/prematch/etape2/split/full_dataset_with_xg.csv` (9,811 matches, 2022-2026)

**Metadata**: JSON file tracks split statistics

---

### **Step 9: EDA2 (Comprehensive Analysis)**

**Purpose**: Deep analysis to guide final cleaning

**File**: `preprocessing/etape2/6_eda/6_eda2_comprehensive_analysis.py`

**Analyses**:

**1. Data Quality**:
- NaN % per feature (severe >50%, moderate 10-50%, light <10%)
- NaN % per match (identify problematic matches >70% NaN)
- Temporal NaN patterns (by year, by league)

**2. Redundancy**:
- Constant columns (nunique ≤ 1)
- High correlation pairs (>0.95)
- Leakage candidates (result-dependent features)

**3. Feature Importance**:
- Correlation with target
- Random Forest importance (300 trees, 10K sample)
- Identify useless features (importance < 0.001)

**4. Distributions**:
- Outliers (extreme ranges)
- Skewness (candidates for log transform)
- Imbalanced features (>90% same value)

**5. Temporal Patterns**:
- Data completeness by year
- Feature stability (variance by year)
- Drift detection

**6. League Analysis**:
- Completeness by league
- Home advantage by league

**Output**: `data/eda/etape2/{dataset_name}/eda_results.json`

**Generates Recommendations**:
```json
{
  "features_to_drop": [...],
  "matches_to_drop": [...],
  "imputation_strategy": {
    "light": [...],
    "moderate": [...]
  },
  "transformations": [
    ["clip", [list_of_features]],
    ["log", [list_of_features]]
  ]
}
```

---

### **Step 10: Advanced Cleaning**

**Purpose**: Apply EDA2 recommendations

**File**: `preprocessing/etape2/7_cleaning.py`

**Steps**:

**1. Temporal Filtering**:
```python
df = df[df['year'] >= 2017]  # Drop 2015-2016
```

**2. Drop Problematic Features**:
- Severe NaN (>50%): Drop immediately
- Constant columns: Drop
- High correlation (>0.95): Drop one of pair
- Low importance (<0.001): Drop
- Leakage suspects: Drop (e.g., `bigchancescored`, `fail_to_score_rate`)

**3. Drop Problematic Matches**:
```python
nan_per_match = df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
df = df[nan_per_match <= 0.70]  # Keep only matches with <70% NaN
```

**4. Imputation MIX**:

**Light NaN (<10%)** - Median by league:
```python
for col in light_nan_cols:
    df[col] = df.groupby('league')[col].transform(lambda x: x.fillna(x.median()))
```

**Moderate NaN (10-30%)** - Smart strategy:
- **Independent features** → Median by league
- **Correlated SAFE features** → KNN (n=5, distance-weighted)

```python
safe_for_knn = [
    col for col in moderate_nan 
    if any(keyword in col.lower() for keyword in [
        'shotsongoal', 'shotsinsidebox', 'corners',
        'possession', 'tackles', 'interceptions',
        # Process stats, NOT results
    ])
]

imputer = KNNImputer(n_neighbors=5, weights='distance')
df[safe_for_knn] = imputer.fit_transform(df[safe_for_knn + context_cols])
```

**NEVER KNN on**: `goals`, `bigchancescored`, `fail_to_score_rate` (result-dependent)

**5. Transformations**:

**Clip outliers** (1st/99th percentiles):
```python
for col in outlier_features:
    p01, p99 = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lower=p01, upper=p99)
```

**Log transform** skewed features (|skew| > 2):
```python
for col in skewed_features:
    if df[col].min() < 0:
        df[f'{col}_log'] = np.sign(df[col]) * np.log1p(abs(df[col]))
    else:
        df[f'{col}_log'] = np.log1p(df[col])
    # Keep original for model comparison
```

**6. Validate 2017** (check if sufficient quality post-imputation):
```python
nan_pct_2017 = df_2017[features].isnull().sum().sum() / (len(df_2017) * len(features))
if nan_pct_2017 > 0.15:  # >15% NaN threshold
    df = df[df['year'] != 2017]  # Drop 2017
```

**Output**: `data/clean/prematch/etape3/full_dataset_{no_xg/with_xg}_clean.csv`

---

### **Step 11: Feature Recovery**

**Purpose**: Recover important features dropped by correlation

**File**: `preprocessing/etape2/8_feature_recovery.py`

**Problem**: Correlation-based dropping removed important features

**Critical features to recover**:
- `elo_diff` (top importance, dropped due to correlation with `expected_prob`)
- `diff_goal_diff_10` (top 6-8 importance)
- `home_form_10`, `away_form_10` (different from `win_rate`)
- `away_attack_defense_ratio` (individual info)

**Strategy**:
```python
def recover_critical_features(df_clean, df_original):
    critical_features = ['elo_diff', 'diff_goal_diff_10', ...]
    
    for feature in critical_features:
        if feature in df_original.columns and feature not in df_clean.columns:
            df_clean[feature] = df_original.loc[df_clean.index, feature]
```

**Output**: `data/clean/prematch/etape3/full_dataset_{no_xg/with_xg}_clean_v2.csv`

**Final Statistics**:
- **no_xG**: 36,940 matches, 137 features, 0% NaN, 2017-2026
- **xG**: 9,811 matches, 140+ features, 0% NaN, 2022-2026

---

## 🤖 Model Training Pipeline

### **Phase 1: Feature Selection**

**File**: `models/xgboost/core/step2b_optimization.py`

**Test 3 configurations**:
```python
configs = {
    'Minimal': XGBOOST_FEATURES_MINIMAL (12 features),
    'Medium': XGBOOST_FEATURES_MEDIUM (45 features),
    'Full': XGBOOST_FEATURES_NO_XG_V1 (137 features)
}
```

**Method**:
- Train simple XGBoost on each configuration
- Evaluate on CV set (Log Loss + Brier Score)
- Compare performance

**Result**: Full configuration best (CV Brier: 0.6049)

---

### **Phase 2: Hyperparameter Optimization**

**Framework**: Optuna with TimeSeriesSplit

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 15.0),
        # ... other params
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    brier_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train[train_idx], y_train[train_idx])
        probs = model.predict_proba(X_train[val_idx])
        
        # Brier Score
        y_val_onehot = np.zeros((len(y_val), 3))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        brier = np.mean(np.sum((probs - y_val_onehot) ** 2, axis=1))
        brier_scores.append(brier)
    
    return np.mean(brier_scores)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)
```

**Key Choices**:
- **Brier Score** as metric (better for probability calibration)
- **TimeSeriesSplit** (respects temporal order, 5 folds)
- **100 trials** (200 overfits optimization)

**Output**: Best parameters saved to JSON

---

### **Phase 3: Ensemble Training**

**Strategy**: Train 5 models with different random seeds

```python
models = []
for seed in [42, 43, 44, 45, 46]:
    params['random_state'] = seed
    model = XGBoostImproved(features, params)
    model.fit(train_df, eval_set=[(cv_df, y_cv)], early_stopping_rounds=50)
    models.append(model)
```

**Prediction**: Average probabilities
```python
probs = np.mean([m.predict_proba(df) for m in models], axis=0)
```

**Benefits**:
- Reduces variance (different initializations)
- More robust predictions
- Less overfitting

---

### **Phase 4: Isotonic Calibration**

**Purpose**: Improve probability estimates

**Data**: Train+CV combined (more robust than CV alone)

```python
class ManualCalibratedEnsemble:
    def fit(self, X, y):
        # Average predictions from all models
        probs_avg = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        
        # Calibrate each class separately
        for class_idx in range(3):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs_avg[:, class_idx], (y == class_idx))
            self.calibrators[class_idx] = calibrator
    
    def predict_proba(self, df):
        # Average predictions
        probs_avg = np.mean([m.predict_proba(df) for m in self.models], axis=0)
        
        # Calibrate
        probs_calibrated = np.column_stack([
            self.calibrators[i].predict(probs_avg[:, i]) for i in range(3)
        ])
        
        # Renormalize
        return probs_calibrated / probs_calibrated.sum(axis=1, keepdims=True)
```

**Output**: Ensemble model with calibration saved to `models/saved/experiments/`

---

## 💰 Value Betting Pipeline

### **Step 1: Load Model & Data**

```python
model_data = joblib.load('models/saved/experiments/xgboost_optimized.pkl')
ensemble = model_data['ensemble']
features = model_data['features']

test_df = load_test_data()
```

---

### **Step 2: Generate Predictions**

```python
X_test = test_df[features]
probs = ensemble.predict_proba(test_df)  # Calibrated probabilities

test_df['prob_home'] = probs[:, 0]
test_df['prob_draw'] = probs[:, 1]
test_df['prob_away'] = probs[:, 2]
```

---

### **Step 3: Compute Expected Value**

```python
def compute_ev(prob, odds):
    """Expected Value = (prob * odds) - 1"""
    return (prob * odds) - 1

test_df['ev_home'] = compute_ev(test_df['prob_home'], test_df['odds_home'])
test_df['ev_draw'] = compute_ev(test_df['prob_draw'], test_df['odds_draw'])
test_df['ev_away'] = compute_ev(test_df['prob_away'], test_df['odds_away'])
```

---

### **Step 4: Filter Value Bets**

**Pure ROI Strategy**:
```python
value_bets = test_df[
    ((test_df['ev_home'] > 0.10) & (test_df['prob_home'] > 0.55)) |
    ((test_df['ev_away'] > 0.10) & (test_df['prob_away'] > 0.55))
]
# Excludes draws (too random)
```

**Volume Strategy**:
```python
value_bets = test_df[
    ((test_df['ev_home'] > 0.05) & (test_df['prob_home'] > 0.45)) |
    ((test_df['ev_away'] > 0.05) & (test_df['prob_away'] > 0.45))
]
# Still excludes draws
```

---

### **Step 5: Rank & Export**

```python
# Rank by EV
value_bets = value_bets.sort_values('ev_max', ascending=False)

# Export
value_bets.to_csv('results/value_bets.csv', index=False)
```

---

## 🔧 Key Technical Decisions

### **1. Why Playwright API Interception over BeautifulSoup?**
- ✅ SofaScore loads data via JavaScript API calls
- ✅ Can't parse HTML (no data in HTML)
- ✅ Must intercept network responses
- ✅ Captures JSON directly from internal APIs

### **2. Why TimeSeriesSplit over KFold?**
- ✅ Respects temporal order
- ✅ Prevents future data leakage
- ✅ Realistic evaluation (train on past, test on future)

### **3. Why Brier Score over Log Loss?**
- ✅ Better for probability calibration
- ✅ Directly measures squared error in probabilities
- ✅ More relevant for betting (accurate probability estimates)

### **4. Why Ensemble over Single Model?**
- ✅ Reduces variance (different initializations)
- ✅ More robust predictions
- ✅ Lower overfitting (2.13% vs 5.36%)

### **5. Why Isotonic over Platt Scaling?**
- ✅ More flexible (non-parametric)
- ✅ Better for non-monotonic relationships
- ✅ Works well with ensemble averaging

### **6. Why KNN Only on Safe Features?**
- ✅ Avoid leakage from result-dependent features
- ✅ `shotsongoal`, `corners` safe (process, not outcome)
- ❌ `goals`, `bigchancescored` unsafe (direct results)

### **7. Why Drop 2015-2016?**
- ✅ Insufficient stats quality (<50% completeness)
- ✅ Missing key features (possession, advanced metrics)
- ✅ Better to have less data with higher quality

---

## 📊 Performance Monitoring

### **Metrics Tracked**

**Classification**:
- Accuracy (per outcome: Home/Draw/Away)
- Log Loss (overall prediction quality)
- Brier Score (probability calibration)

**Betting**:
- ROI (return on investment)
- Profit (absolute gains)
- Number of bets placed
- Win rate

**Overfitting**:
- Train-Test accuracy gap
- Train-Test Log Loss gap
- Train-Test Brier gap

### **Evaluation Sets**

- **Train**: 60% of data (oldest matches)
- **CV**: 20% of data (middle period)
- **Test**: 20% of data (most recent)

Temporal split ensures realistic evaluation.

---

## 🚀 Deployment Architecture

**Current**: Local execution
**Future**: See [Future Work](future_work.md)

**Planned Architecture**:
```
┌──────────────┐
│   Scraper    │ → Cron job (daily)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Database    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ML Pipeline │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Dashboard   │
└──────────────┘
```

---

*Last Updated: February 2026*