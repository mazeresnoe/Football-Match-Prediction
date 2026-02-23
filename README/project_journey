# 📖 Project Journey

> **The complete story of building a football prediction system from scratch**

---

## 🎬 Introduction

This document chronicles the entire development journey of the Football Match Prediction System, from initial idea to final implementation. It covers challenges faced, solutions implemented, and lessons learned along the way.

**Timeline**: Started in 2025, active development through February 2026

**Goal**: Build an end-to-end ML system that can predict the outcome of a football game (1N2), and then identify profitable betting opportunities on selective matches.

---

## 📅 Development Phases

### **Phase 1: Data Collection**

#### Initial Vision
- Scrape comprehensive football match statistics from multiple sources (Game's stats from Sofascore and Odd's stats from football-data.co.uk)
- Cover major European leagues with 5+ years of historical data
- Include both basic stats and advanced metrics (xG, possession, shots)

#### Challenges Faced

**Challenge 1: SofaScore API Response Interception**

SofaScore doesn't expose direct APIs - all data comes from internal API calls that need to be intercepted.

```python
# ❌ This doesn't work - no HTML parsing possible
response = requests.get(url)
soup = BeautifulSoup(response.content)
# Page is empty! All data loaded via JavaScript

# ✅ Solution: Intercept API responses
page.on("response", handle_response)
def handle_response(response):
    if "api/v1/event" in response.url:
        data = response.json()
        # Extract data from API response
```

**Solution**: Playwright with response interception
- Capture JSON responses from internal APIs
- Extract data directly from API calls
- **3 separate scrapers**:
  - `round_scrapper_2.py`: Get event IDs for every match per league/season
  - `match_stats_scraper.py`: Get detailed stats thanks to the events IDs
  - `scrape_odds.py` : Get the odds for the bookmaker

**Challenge 2: Team Name Inconsistencies (MAJOR)**

Sofascore and football-data use different team names  - **300+ variations** to map:
- "Manchester United" vs "Man United" vs "Man Utd"
- "PSV Eindhoven" vs "PSV"
- "Athletic Club" vs "Athletic Bilbao"
- "Borussia M'gladbach" vs "M'gladbach" vs "Mönchengladbach"

**Solution**: Manual team mapping (`src/scraping/odds/team_mapping_complete.py`)
- Created comprehensive mapping dictionary
- Standardized names across all data sources
- Mapped 300+ team name variations manually

Another mapping is done for the season's id for every tournament, tournament's id, and maximum amount of games per league for a given season from sofascore (`src\utils\mapping\map.py`).

**Challenge 3: Data Availability Timeline**

Not all statistics available for all years:
- **2015-2016**: Basic stats only (goals, shots) - **INSUFFICIENT**
- **2017-2021**: Enhanced stats added
- **2022+**: Full xG metrics introduced

**Solution**: Dual dataset approach + temporal filtering
- **Raw scraping**: 2015-2026 (42,911 matches)
- **Cleaning drops 2015-2016**: Not enough quality stats
- **no_xG dataset**: 2017-2026 (36,940 matches) - enhanced stats without xG
- **xG dataset**: 2022-2026 (9,811 matches) - full xG metrics

**Challenge 4: Progress Tracking & Resume**

Scraping 42K+ matches takes hours - need to handle interruptions.

**Solution**: Progress tracking system
```python
# Save progress after each round
progress = {
    "scraped_events": list(scraped_events),
    "failed_events": list(failed_events),
    "last_update": datetime.now().isoformat()
}
save_progress(league, season, progress)
```

- JSON file per league/season tracking scraped event IDs
- Resume from last checkpoint on interruption
- Skip already-scraped matches
- Retry failed events with exponential backoff

#### Results Achieved
- ✅ **42,911 raw matches** scraped successfully
- ✅ **14 competitions** covered
- ✅ **11 years** of data (2015-2026)
- ✅ **Automated scraping pipeline** with progress tracking
- ✅ **300+ team name variations** mapped

---

### **Phase 2: Basic Data Cleaning & EDA **

#### Cleaning Pipeline (8 Steps)

**Step 1: Raw Merger** (`1_raw_merger.py`)
- Combine all league CSVs into single dataset
- Normalize column names (lowercase, underscores)
- Normalize team names
- Add metadata (league, season, source_file)

**Step 2: Basic Cleaning** (`2_cleaning.py`)
- Drop 100% empty columns
- Remove canceled/postponed matches
- Convert date formats
- Remove duplicates
- Reorder columns (ID cols first)

**Result**: `post_match_clean.csv` (42,911 matches)

#### Challenges Faced

**Challenge 1: Messy Raw Data**

Issues discovered:
- Missing values
- Duplicate matches (same match scraped multiple times)
- Encoding errors (special characters in team names)
- Inconsistent date formats
- Some matches missing critical info (scores, teams)

**Solution**: Multi-step cleaning pipeline
```python
# Step 1: Remove duplicates based on date + teams
df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'])

# Step 2: Standardize dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Step 3: Handle missing values strategically
# - Drop matches missing critical info (result, teams, date)
# - Impute stats later in feature recovery phase
```

**Challenge 2: Data Quality Varies by League**

- Top 5 leagues: High quality, complete data
- Second divisions: More missing values
- European competitions: Inconsistent formats (group stage vs knockouts)

**Solution**: League-specific cleaning rules
- Stricter requirements for top leagues
- More lenient for second divisions (accept more missingness)
- Custom handling for tournament formats (Champions League structure)

**Challenge 3: Temporal Data Leakage Prevention**

Must ensure no "future" information leaks into training:
- Team form must use ONLY past matches
- Rolling averages must respect chronological order
- Elo updates must be sequential

**Solution**: Strict temporal validation
- Sorted all data by date BEFORE any feature engineering
- Implemented chronological feature computation with lookback windows
- Added assertions to catch temporal leaks
- Validated no match uses stats from future matches

#### Results Achieved
- ✅ **36,940 clean matches** for no_xG dataset (after dropping 2015-2016)
- ✅ **9,811 clean matches** for xG dataset (2022+)
- ✅ **Zero temporal leakage** confirmed through validation
- ✅ **Quality thresholds** met for all leagues

---

### **Phase 3: Feature Engineering**

#### Strategy

Create features capturing:
1. **Team Strength**: Elo ratings, win rates, goal differentials
2. **Recent Form**: Last 5/10 match performance
3. **Head-to-Head**: Historical matchups between teams
4. **League Context**: Home advantage, league difficulty
5. **Advanced Metrics**: xG, possession, shots on target
6. **Momentum**: Short-term vs long-term trends
7. **Efficiency**: Conversion rates, defensive solidity

#### Challenges Faced

**Challenge 1: Cold Start Problem**

New teams entering dataset have no historical data:
- Promoted teams from lower divisions
- Teams in first season of tracking

**Solution**: Hierarchical initialization
```python
if team_has_history:
    elo = historical_elo
elif team_in_lower_division:
    elo = lower_division_average + promotion_bonus
else:
    elo = league_average  # Default: 1500
```

**Challenge 2: Feature Explosion**

Initial feature engineering created 500+ features:
- Many redundant (correlated >0.95)
- Some noisy (random, low importance)
- Computational overhead

**Solution**: Feature selection + regularization
- Tested Minimal (12), Medium (45), Full (137) configurations
- Found Full configuration optimal (best Log Loss/Brier)
- Regularization handles redundancy better than manual pruning
- XGBoost automatically learns feature importance

**Challenge 3: Missing xG for Historical Matches**

xG only available since 2022:
- Can't compute xG-based features for 2017-2021
- Need to maintain two separate datasets

**Solution**: Dual dataset strategy with smart handling
- **no_xG**: Focus on traditional stats, larger sample (36,940 matches)
- **xG**: Include advanced metrics, smaller sample (9,811 matches)
- Created `has_xg_stats` completeness score
- xG features = NaN for old matches (model handles missing)
- Train separate models, use no_xG as primary

**Challenge 4: Feature Engineering in 2 Passes**

Too many features to create in one pass:

**Solution**: V1 + V2 approach

**V1 (Basic)** - `3_feature_eng_v1_basic.py`:
- Elo ratings (1500 initial, K=20, home advantage +100)
- Form metrics (last 5/10 matches)
- Head-to-head stats (last 5/10 encounters)
- Rolling averages (goals, shots, possession)
- Rest days (days since last match)

**V2 (Advanced)** - `4_feature_eng_v2_advanced.py`:
- Temporal features (month, season stage, fatigue)
- Elo interactions (elo × form, elo × goals)
- Momentum (short-term trend vs long-term)
- Efficiency ratios (conversion rates)
- Balance features (attack/defense ratio)
- Pressure metrics (corners, final third entries)
- Big chance features (creation + conversion)
- H2H advanced (win rates, goal differences)
- Stability metrics (form volatility)
- xG features (overperformance, momentum)
- Completeness scores (data quality indicators)

#### Results Achieved
- ✅ **270+ features** engineered (V1 + V2)
- ✅ **3 feature configurations** tested
- ✅ **Dual dataset** approach working
- ✅ **No temporal leakage** in features (validated)
- ✅ **xG handling** (NaN for unavailable data, therefore dropped for no_xg)

---

### **Phase 4: EDA2 & Advanced Cleaning**

#### EDA2 Comprehensive Analysis (`6_eda2_comprehensive_analysis.py`)

**Analyses Performed**:
1. **Data Quality**: NaN patterns, problematic matches
2. **Redundancy**: Constant columns, high correlations (>0.95)
3. **Feature Importance**: Random Forest + correlation with target
4. **Distributions**: Outliers, skewness, imbalanced features
5. **Temporal Patterns**: Data drift, completeness by year/league
6. **League Analysis**: Home advantage, quality differences

**Key Findings**:
- 🔴 **Severe NaN** (>50%): Some features unusable
- 🔴 **High correlation** (>0.95): 40+ redundant features
- 🔴 **Leakage suspects**: Features like `bigchancescored` (result-based)
- 🟡 **Moderate NaN** (10-30%): Need smart imputation
- 🟢 **Top features**: elo_diff, form metrics, goal differentials

#### Advanced Cleaning (`7_cleaning.py`)

**Strategy**: Based on EDA2 recommendations

**Steps**:
1. Drop years 2015-2016 (insufficient stats)
2. Drop problematic features (severe NaN, constant, high corr, leakage)
3. Drop matches with >70% NaN
4. **Imputation MIX**:
   - Light NaN (<10%): Median by league
   - Moderate NaN (10-30%): 
     - Independent features → Median by league
     - Correlated safe features → KNN (n=5, distance-weighted)
5. Transformations:
   - Clip outliers (1st/99th percentiles)
   - Log transform skewed features (|skew| > 2)
6. Validate 2017 quality (check NaN % post-imputation)

**KNN Imputation - Safe Features Only**:
```python
safe_for_knn = [
    'shotsongoal', 'shotsinsidebox', 'corners', 
    'possession', 'tackles', 'interceptions',
    # Process stats, NOT results
]
# NEVER on: goals, bigchancescored, fail_to_score_rate
```

#### Feature Recovery (`8_feature_recovery.py`)

**Problem**: Correlation-based dropping removed important features

**Solution**: Recover critical features
- `elo_diff` (top importance, dropped due to correlation with `expected_prob`)
- `diff_goal_diff_10` (top 6-8 importance)
- `home_form_10`, `away_form_10` (different from `win_rate`)

**Strategy**: Keep BOTH correlated features when both important

#### Results Achieved
- ✅ **Final datasets** ready for modeling
- ✅ **0% NaN** after imputation
- ✅ **137 features** (Full configuration)
- ✅ **No leakage** (removed result-dependent features)
- ✅ **Balanced** (kept important features despite correlation)

---

### **Phase 5: Baseline Models**

#### Models Tested

**1. Simple Elo**
- Basic Elo rating system (K=20, home+100)
- Probability from Elo difference
- No additional features

**Results**:
```
Train: 47.71% accuracy, 1.1635 Log Loss, 0.6697 Brier Score
CV: 48.80% accuracy, 1.1463 Log Loss, 0.6591 Brier Score
Test: 48.90% accuracy, 1.1448 Log Loss, 0.6589 Brier Score
```

**Analysis**: Too simple, underfit and doesn't capture form or context.

**2. Logistic Regression**
- Linear model with 137 features
- L2 regularization (C=1.0)
- StandardScaler + SimpleImputer

**Results**:
```
Train: 48.68% accuracy, 1.0226 Log Loss, 0.6132 Brier Score
CV: 49.49% accuracy, 1.0156 Log Loss, 0.6085 Brier Score
Test: 49.15% accuracy, 1.0181 Log Loss, 0.6097 Brier Score
```

**Analysis**: Surprisingly competitive! But **underfits** (test better than train = model too simple).

**3. Bookmaker Baseline**
- Convert odds to probabilities
- Two methods: Simple and Proportional margin removal

**Results**:
```
Train: 49.43% accuracy, 1.0092 Log Loss, 0.6034 Brier Score
CV: 50.15% accuracy, 1.0071 Log Loss, 0.6028 Brier Score
Test: 49.95% accuracy, 1.0016 Log Loss, 0.5993 Brier Score
```

**Analysis**: Very strong baseline. Bookmakers have insider info we don't.

#### Key Insights

1. **Bookmakers are very good** - They remain the best predictor
2. **Simple models competitive** - Diminishing returns from complexity
3. **Football is inherently noisy** - ~50% accuracy ceiling for everyone
4. **ROI all negative** - Can't beat bookmakers by betting on everything

---

### **Phase 6: XGBoost Development**

#### Initial Attempt: XGBoost Baseline

**First model**: Default parameters, Full features

**Results**:
```
Train:  54.71% accuracy, 0.9350 Log Loss, 0.5066 Brier score
CV:     49.82% accuracy, 1.0097 Log Loss, 0.6201 Brier score
Test:   49.35% accuracy, 1.0141 Log Loss, 0.6203 Brier score
```

**Analysis**: Clear overfitting! Model memorizing training data.

#### Problem Analysis

**Why overfitting?**
1. ❌ No hyperparameter tuning (using defaults)
2. ❌ Simple train/cv/test split (no temporal CV)
3. ❌ Optimizing wrong metric (Log Loss on single split)
4. ❌ No calibration (probabilities not well-calibrated)
5. ❌ Single model (high variance, seed-dependent)

---

### **Phase 7: Optimization & Overfitting Reduction**

#### Strategy: Multi-Pronged Attack

**1. Feature Selection First**

📊 Minimal (12 features)

Train: 51.94% accuracy, 0.9714 Log Loss, 0.5797 Brier Score
CV: 49.31% accuracy, 1.0190 Log Loss, 0.6104 Brier Score
Test: 49.26% accuracy, 1.0208 Log Loss, 0.6116 Brier Score

📊 Medium (45 features)

Train: 55.83% accuracy, 0.9282 Log Loss, 0.5506 Brier Score
CV: 49.77% accuracy, 1.0155 Log Loss, 0.6080 Brier Score
Test: 49.16% accuracy, 1.0171 Log Loss, 0.6088 Brier Score

📊 Full (137 features)

Train: 60.98% accuracy, 0.8760 Log Loss, 0.5155 Brier Score
CV: 49.70% accuracy, 1.0117 Log Loss, 0.6054 Brier Score
Test: 49.26% accuracy, 1.0157 Log Loss, 0.6080 Brier Score
```

**Result**: Full configuration best, use for optimization.

**2. Hyperparameter Optimization (Optuna)**

Used Optuna with:
- **100 trials** (200 overfits optimization itself)
- **TimeSeriesSplit** (5 folds, respects chronological order)
- **Brier Score** as metric (better for probability calibration than Log Loss)

Search space:
```python
{
    'n_estimators': 100-800 (step 100),
    'max_depth': 3-6,
    'learning_rate': 0.01-0.1 (log scale),
    'subsample': 0.6-0.9,
    'colsample_bytree': 0.6-0.9,
    'min_child_weight': 3-15,
    'gamma': 0-2.0,
    'reg_alpha': 0-5.0,
    'reg_lambda': 5.0-15.0,  # Strong L2 regularization
}
```

**3. Ensemble Multi-Seed**

Train 5 models with different random seeds:
- Seeds: 42, 43, 44, 45, 46
- Average predictions (reduces variance)
- More robust than single model

**4. Isotonic Calibration**

Calibrate probabilities on Train+CV combined:
- More data = better calibration (vs CV only)
- Separate calibrator per class (Home/Draw/Away)
- Renormalize to sum to 1

Implementation:
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
```

#### Results Achieved

**Final XGBoost Ensemble**:
```
Train:  51.03% accuracy, 0.9951 Log Loss, 0.5958 Brier
CV:     49.93% accuracy, 1.0100 Log Loss, 0.6049 Brier
Test:   48.98% accuracy, 1.0210 Log Loss, 0.6089 Brier

Overfitting Gap: 2.13%, way less than before
```

**Improvements**:
- ✅ Overfitting
- ✅ Best Brier Score among ML models (0.6089)
- ✅ Robust ensemble (low variance across seeds)
- ✅ Well-calibrated probabilities (isotonic on Train+CV)

---

### **Phase 8: Value Betting Strategy**

#### Challenge: Global ROI Still Negative

Even best model has -3.45% ROI when betting everything.

**Why?**
- Bookmakers still better overall
- We lack insider information
- Calculation of the marge is not exact

#### Solution: Selective Betting

**Strategy**: Only bet when we have edge

Test 532 combinations:
```python
for ev_threshold in [2%, 4%, ..., 20%]:
    for confidence_threshold in [35%, 40%, ..., 65%]:
        for bet_type in ['all', 'no_draw', 'home_away_only']:
            compute_roi()
```

**Results**:

**Pure ROI Strategy**:
```
- EV Threshold: 10%
- Confidence: 55%
- Bet Type: No Draw
- Result: 24 bets/year, 66.67% win rate, +16.62% ROI
```

**Volume Strategy**:
```
- EV Threshold: 5%
- Confidence: 45%
- Bet Type: No Draw
- Result: 198 bets/year, 51.01% win rate, +7.97% ROI
```

**Key Insight**: Volume strategy generates **4x more profit** despite lower ROI (diversification).

---

## 🎯 Final Results Summary

### Model Performance

| Model | Log Loss | Brier | Overfitting | Note |
|-------|----------|-------|-------------|------|
| Bookmaker | 1.0016 | 0.5993 | N/A | Insider info |
| **XGBoost Ensemble** | 1.0210 | **0.6089** | **2.13%** | Best ML |
| LogReg | 1.0181 | 0.6097 | N/A | Underfits |

### Betting Performance

| Strategy | ROI | Bets/Year | Profit/Year (€10/bet) |
|----------|-----|-----------|----------------------|
| Pure ROI | +16.62% | 24 | +40€ |
| **Volume** | +7.97% | 198 | **+158€** |

---

## 💡 Key Takeaways

### What Worked ✅

1. **Playwright API interception** - Capture JSON responses
2. **Manual team mapping** - 300+ variations handled
3. **Dual dataset approach** - Separate no_xG (primary) and xG (experimental)
4. **Temporal validation** - TimeSeriesSplit respects chronological order
5. **8-step cleaning pipeline** - EDA-driven, systematic approach
6. **KNN imputation (safe features only)** - Better than global median
7. **Feature selection BEFORE optimization** - Test Minimal/Medium/Full
8. **Ensemble + Calibration** - Reduces variance, improves probabilities
9. **Selective betting** - Only bet when statistical edge exists

### What Didn't Work ❌

1. **Default hyperparameters** - Led to high overfitting
2. **Single model** - High variance, seed-dependent
3. **Global betting** - Negative ROI, can't beat bookmakers everywhere
4. **Aggressive feature pruning** - Full features better than Minimal
5. **200 Optuna trials** - Overfits optimization, 100 optimal
6. **Calibration on CV only** - Train+CV gives more robust calibration

### Critical Decisions

1. **Drop 2015-2016** - Insufficient stats quality
2. **Focus on no_xG dataset** - More data (36,940 vs 9,811)
3. **Brier Score optimization** - Better than Log Loss in this case
4. **Volume strategy** - Higher total profit than pure ROI
5. **Exclude draws** - Too random to predict reliably
6. **KNN only on safe features** - Avoid leakage from result-dependent features

---

## 🚀 What's Next?

See [Future Work](future_work.md) for detailed roadmap.

**Priority improvements**:
1. Scrape OddsPortal for live/future odds
2. Add Transfermarkt player-level data
3. Expand xG dataset (wait 2-3 years OR scrape historical xG from Understat)
4. Automated betting notifications
5. Production dashboard deployment

---

*Last Updated: February 2026*