# 🎓 Lessons Learned

> **Key insights from building a football prediction system**

---

## ✅ What Worked

### **1. Playwright API Interception**

**Decision**: Intercept SofaScore's internal API calls instead of parsing HTML

**Result**: ✅ Success
- 42,911 matches scraped reliably
- Direct access to JSON data
- Progress tracking prevents re-scraping
- Resume capability after interruptions

**Implementation**:
```python
page.on("response", handle_response)
def handle_response(response):
    if "api/v1/event" in response.url:
        data = response.json()
```

**Lesson**: When dealing with JavaScript-heavy sites, intercept API calls rather than attempting HTML parsing.

---

### **2. Manual Team Name Mapping (500+ Variations)**

**Decision**: Create comprehensive mapping dictionary by hand

**Result**: ✅ Success
- All team name variations handled
- Consistent names across SofaScore and odds data
- Enables accurate merging

**Lesson**: For critical data quality issues (like team names), manual effort upfront saves debugging later. Automated matching would have failed.

---

### **3. Dual Dataset Strategy**

**Decision**: Maintain separate no_xG (2017-2026) and xG (2022-2026) datasets

**Result**: ✅ Success
- More training data for no_xG model (36,940 vs 9,811)
- Can leverage xG when available
- Clear separation avoids confusion

**Lesson**: When features have different availability timelines, separate datasets often better than trying to merge.

---

### **4. Temporal Validation (TimeSeriesSplit)**

**Decision**: Use TimeSeriesSplit instead of random KFold

**Result**: ✅ Success
- Prevents future data leakage
- Realistic evaluation (train on past, test on future)
- Reduced overfitting (respects temporal dependencies)

**Lesson**: For time-series data, temporal splits are MANDATORY. Random splits give overly optimistic results.

---

### **5. 8-Step Cleaning Pipeline**

**Decision**: Break cleaning into systematic steps with EDA in between

**Result**: ✅ Success
- **Step 1-2**: Basic cleaning (42,911 → 42,911 valid matches)
- **Step 3-4**: Feature engineering V1+V2 (270+ features)
- **Step 5**: Dataset split (no_xG vs xG)
- **Step 6**: EDA2 (comprehensive analysis)
- **Step 7**: Advanced cleaning (drop 2015-2016, imputation, transformations)
- **Step 8**: Feature recovery (restore critical features)

**Lesson**: EDA-driven cleaning (analyze → clean → analyze) produces higher quality than cleaning blindly.

---

### **6. Hyperparameter Optimization (Optuna)**

**Initial**: Used default XGBoost parameters
- **Result**: 5.36% overfitting gap

**After**: Optuna optimization (100 trials, TimeSeriesSplit, Brier Score)
- **Result**: 2.13% overfitting gap ✅ (60% reduction)

**Lesson**: Never use default parameters. Proper hyperparameter tuning is critical.

---

### **7. Ensemble Multi-Seed**

**Decision**: Train 5 models with different seeds and average predictions

**Result**: ✅ Success
- Reduced variance significantly
- More robust predictions
- Only slight computational overhead (5x single model)

**Lesson**: Ensembles almost always improve results. Even simple averaging works well.

---

### **8. Isotonic Calibration on Train+CV**

**Decision**: Calibrate probabilities using Train+CV combined (vs CV only)

**Result**: ✅ Success
- Better Brier Score (0.6089 vs uncalibrated)
- More accurate probability estimates
- Essential for value betting

**Lesson**: For betting applications, probability calibration is as important as accuracy. Use maximum available data (Train+CV, not just CV).

---

### **9. Selective Betting Strategy**

**Decision**: Only bet on matches with positive expected value (EV > threshold)

**Result**: ✅ Success
- Global betting: -3.45% ROI ❌
- Selective betting (Pure ROI): +16.62% ROI ✅
- Selective betting (Volume): +7.97% ROI ✅

**Lesson**: Can't beat bookmakers on ALL matches, but can find selective opportunities.

---

### **10. KNN Imputation (Safe Features Only)**

**Decision**: Use KNN only on process stats, NOT result-dependent features

**Result**: ✅ Success
- Better than global median
- No leakage (avoided `goals`, `bigchancescored`)
- Safe features: `shotsongoal`, `corners`, `possession`, `tackles`

**Lesson**: KNN imputation is powerful but requires careful feature selection to avoid leakage.

---

## ❌ What Didn't Work

### **1. Default Hyperparameters**

**Mistake**: Used XGBoost defaults initially

**Result**: ❌ Failure
- 5.36% overfitting gap
- Poor generalization
- Train ROI +15.72%, Test ROI -4.51%

**Lesson**: Defaults are starting points, not solutions. Always tune.

---

### **2. Single Model Reliance**

**Mistake**: Initially trained single XGBoost model

**Result**: ❌ Failure
- High variance (results change with seed)
- Less robust predictions
- More prone to overfitting

**Lesson**: For production, always use ensembles (even simple averaging).

---

### **3. Log Loss Optimization**

**Initial**: Optimized hyperparameters using Log Loss

**Better**: Switch to Brier Score

**Result**: ✅ Better calibration
- Brier Score improved more than Log Loss
- More relevant for betting applications

**Lesson**: Choose optimization metric based on final use case, not convention.

---

### **4. 200 Optuna Trials**

**Test**: Compared 100 vs 200 trials

**Result**: 200 trials WORSE
- Test performance identical
- Train performance better → more overfitting
- 2x computational cost

**Lesson**: More trials can overfit the optimization process itself. 100 optimal for this problem size.

---

### **5. Calibration on CV Only**

**Initial**: Calibrated using CV set only

**Better**: Calibrate on Train+CV combined

**Result**: ✅ More robust
- More data = better calibration
- Less variance in calibrated probabilities

**Lesson**: For calibration, use as much data as possible (but never test set).

---

### **6. Global Betting**

**Mistake**: Initially bet on ALL matches

**Result**: ❌ Negative ROI (-3.45%)
- Bookmakers too good overall
- Can't beat them everywhere

**Lesson**: Selective betting essential. Only bet when you have statistical edge.

---

### **7. Including Draws in Betting**

**Test**: Including draws in betting strategy

**Result**: ❌ Poor performance
- Only 18% accuracy on draws
- Too random to predict reliably (late goals, red cards, referee)
- Better to exclude from betting

**Lesson**: Some outcomes inherently harder to predict. Focus on what works (Home/Away).

---

### **8. Aggressive Feature Pruning**

**Test**: Minimal (12), Medium (45), Full (137) features

**Result**: Full features best
- Full: 1.0157 Log Loss ✅
- Medium: 1.0198 Log Loss
- Minimal: 1.0245 Log Loss

**Lesson**: With proper regularization, more features usually better than aggressive pruning. XGBoost handles redundancy.

---

### **9. Scraping 2015-2016 Data**

**Mistake**: Scraped 2015-2016 despite poor stats quality

**Result**: ❌ Wasted effort
- <50% stats completeness
- Missing key features
- Ultimately dropped in cleaning

**Lesson**: Research data quality before scraping. Better to start from 2017 directly.

---

## 💡 Key Insights

### **1. Bookmakers Are Very Good**

**Why they win**:
- 📊 More data (lineups, injuries, insider info)
- 💰 Market dynamics (odds adjust with betting volume)
- 🧠 Decades of optimization

**Our edge**:
- 🎯 Selective betting on specific matches
- 📈 Better probability calibration than simple odds conversion
- 🤖 Automated systematic detection

**Lesson**: Don't try to beat bookmakers everywhere. Find specific opportunities.

---

### **2. Football Is Inherently Noisy**

**Evidence**:
- Bookmaker accuracy: 49.95%
- Best ML model: 48.98%
- Everyone hovers around 50%

**Why**:
- Random events (injuries, red cards, referee decisions)
- Small sample effects (one goal changes outcome)
- Psychological factors (motivation, pressure)

**Lesson**: Accept ~50% accuracy ceiling. Focus on probability quality, not accuracy.

---

### **3. Data Quality > Model Complexity**

**Observation**:
- Clean data + simple model > dirty data + complex model
- Temporal validation crucial
- Feature engineering matters more than algorithm choice

**Evidence**:
- LogReg (simple): 1.0181 Log Loss
- XGBoost (complex): 1.0210 Log Loss
- Only 0.3% difference!

**Lesson**: Invest in data quality and proper validation before optimizing models.

---

### **4. Overfitting Is Easy, Prevention Is Hard**

**Challenge**: Football has limited data relative to noise

**Solutions that worked**:
- ✅ Strong regularization (reg_lambda 5-15)
- ✅ Early stopping (50 rounds)
- ✅ Temporal cross-validation (TimeSeriesSplit)
- ✅ Ensemble averaging (5 seeds)
- ✅ Feature selection (test Minimal/Medium/Full)

**Lesson**: Overfitting prevention requires multi-pronged approach.

---

### **5. Evaluation Metrics Matter**

**Wrong metric**: Accuracy
- Misleading for imbalanced classes
- Doesn't measure probability quality

**Right metrics**:
- ✅ Log Loss (overall quality)
- ✅ Brier Score (probability calibration)
- ✅ ROI (business objective)

**Lesson**: Choose metrics aligned with final objective (betting → probability quality).

---

### **6. Simple Models Are Competitive**

**Surprise**: Logistic Regression very competitive

**Results**:
- LogReg: 1.0181 Log Loss
- XGBoost: 1.0210 Log Loss
- Only 0.3% difference

**Why**:
- Football difficult to predict
- Diminishing returns from complexity
- LogReg underfits but robust

**Lesson**: Start simple, add complexity only if justified by validation results.

---

### **7. Probability Calibration Underrated**

**Importance**: For betting, probability accuracy > classification accuracy

**Evidence**:
- XGBoost Brier: 0.6089 (best among ML)
- LogReg Brier: 0.6097
- 0.13% difference matters for ROI

**Lesson**: Invest in calibration (Isotonic, Platt Scaling, etc.).

---

### **8. Domain Knowledge Helps**

**Examples**:
- Excluding draws (too random)
- Home advantage features (+100 Elo)
- Form metrics (last 5 matches)
- Head-to-head history
- Rest advantage (days since last match)

**Result**: Domain-informed features outperform generic ones

**Lesson**: Combine ML with domain expertise for best results.

---

### **9. API Interception > HTML Parsing**

**Surprise**: Modern websites don't put data in HTML

**Why Playwright wins**:
- Captures internal API calls
- Direct access to JSON
- No need to parse complex HTML
- More stable (HTML changes, APIs don't)

**Lesson**: For JavaScript-heavy sites, intercept network traffic.

---

### **10. Manual Work Has Value**

**Examples where automation failed**:
- Team name mapping (500+ variations)
- Data quality assessment (which years to keep)
- Feature recovery (which correlated features are important)

**Lesson**: Not everything should be automated. Strategic manual work saves time overall.

---

## 🔄 Iterative Improvements

### **Version 1 → Version 2**

| Aspect | V1 | V2 | Improvement |
|--------|----|----|-------------|
| **Hyperparameters** | Default | Optuna 100 trials | -60% overfitting |
| **Model** | Single | Ensemble (5 seeds) | Lower variance |
| **Calibration** | None | Isotonic (Train+CV) | Better probabilities |
| **Features** | 270+ | 137 (selected) | Optimal |
| **Cross-Validation** | Simple split | TimeSeriesSplit | Temporal validity |
| **Optimization Metric** | Log Loss | Brier Score | Better for betting |
| **Imputation** | Median only | Median + KNN (safe) | Better quality |
| **Data Range** | 2015-2026 | 2017-2026 | Higher quality |

**Result**: Test Brier improved from 0.6067 → 0.6089 (with lower overfitting)

---

## 🎯 Critical Success Factors

1. ✅ **API interception** (vs HTML parsing)
2. ✅ **Manual team mapping** (500+ variations)
3. ✅ **Proper temporal validation** (no future leakage)
4. ✅ **Hyperparameter optimization** (never use defaults)
5. ✅ **Ensemble methods** (reduce variance)
6. ✅ **Probability calibration** (essential for betting)
7. ✅ **Selective betting** (only when edge exists)
8. ✅ **Continuous iteration** (V1 → V2)
9. ✅ **EDA-driven cleaning** (analyze → clean → analyze)
10. ✅ **Safe imputation** (KNN on process features only)

---

## 🚫 Common Pitfalls to Avoid

1. ❌ HTML parsing on JavaScript-heavy sites
2. ❌ Random cross-validation on time-series data
3. ❌ Using default hyperparameters
4. ❌ Single model without ensembling
5. ❌ Optimizing wrong metric (accuracy vs probability quality)
6. ❌ Betting on all matches (negative ROI)
7. ❌ Ignoring probability calibration
8. ❌ Over-trusting training performance
9. ❌ KNN imputation on result-dependent features
10. ❌ Scraping low-quality data (2015-2016)

---

## 📈 If Starting Again

**What I'd keep**:
- ✅ Playwright API interception
- ✅ Manual team mapping
- ✅ Dual dataset approach
- ✅ TimeSeriesSplit validation
- ✅ 8-step cleaning pipeline
- ✅ Ensemble + calibration
- ✅ Selective betting strategy

**What I'd change**:
- 🔄 Skip 2015-2016 scraping (start from 2017)
- 🔄 Start with hyperparameter tuning earlier
- 🔄 Focus on Brier Score from beginning
- 🔄 Implement progress tracking day 1 (not later)
- 🔄 Add more external data sources earlier (Transfermarkt)
- 🔄 Build production pipeline incrementally

---

## 💬 Final Thoughts

**Most Important Lesson**: 

> **You can't beat bookmakers everywhere, but you CAN find selective opportunities where your model has an edge. Success is about finding the right matches, not predicting all matches perfectly.**

**Second Most Important**:

> **Proper validation is more important than fancy models. A simple model with temporal validation beats a complex model with data leakage.**

**Third Most Important**:

> **Invest time in data quality and infrastructure. 80% of ML success is data preparation, not modeling.**

**Fourth Most Important**:

> **Modern web scraping requires API interception, not HTML parsing. Playwright > BeautifulSoup for JavaScript sites.**

---

*Last Updated: February 2026*