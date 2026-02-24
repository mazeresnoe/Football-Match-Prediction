# ⚠️ Current Limitations

> **Known constraints and areas for improvement**

---

## 📊 Data Limitations

### **1. Limited xG Historical Data**

**Issue**: xG metrics only available since 2022

**Impact**:
- xG dataset: Only 9,811 matches
- no_xG dataset: 36,940 matches (2017-2026)
- Can't train robust xG model yet

**Why**: SofaScore introduced xG tracking in 2022

**Solution**: Wait 2-3 more years for larger xG dataset, or scrape historical xG from other sources

**Severity**: 🟡 Medium

---

### **2. Historical Odds Only**

**Issue**: Only have historical odds, not live or future odds

**Current Source**: football-data.co.uk
- ✅ Historical odds after match completion
- ❌ No live odds during match
- ❌ No future odds for upcoming matches

**Impact**:
- Can backtest strategies ✅
- Can't place real bets automatically ❌
- Can't track odds movements ❌

**Solution**: Scrape OddsPortal or similar for live/future odds

**Severity**: 🔴 High (for production use)

---

### **3. Missing Player-Level Data**

**Issue**: Only team-level statistics, no individual player data

**What's Missing**:
- Player market values
- Individual player form
- Lineup information
- Injury status

**Impact**: Missing potentially predictive features

**Solution**: Integrate Transfermarkt data

**Severity**: 🟡 Medium

---

### **4. No Weather Data**

**Issue**: Weather can impact match outcomes (especially wind, rain)

**Missing**:
- Temperature
- Wind speed
- Precipitation
- Playing conditions

**Impact**: Unaccounted external factor

**Solution**: Integrate weather API (OpenWeatherMap)

**Severity**: 🟢 Low (minor impact)

---

### **5. Limited Competitions**

**Current**: 14 competitions (top 5 leagues + second divisions + European)

**Missing**:
- Lower divisions (3rd tier, etc.)
- Other European leagues (Eredivisie, Primeira Liga)
- South American leagues
- Asian leagues

**Impact**: Limited market coverage

**Solution**: Expand scraping to more leagues (requires more infrastructure)

**Severity**: 🟢 Low (current coverage sufficient)

---

## 🤖 Model Limitations

### **1. Bookmaker Probabilities Are Estimates**

**Issue**: We estimate bookmaker probabilities from odds

**Method**:
```python
# Proportional margin removal
total_implied = 1/odds_home + 1/odds_draw + 1/odds_away
prob_home = (1/odds_home) / total_implied
```

**Reality**: Bookmakers have true ML model outputs we don't see

**Impact**: 
- Baseline comparison not entirely fair
- Our "bookmaker" baseline is approximation

**Solution**: Accept limitation (can't access true bookmaker models)

**Severity**: 🟡 Medium (acknowledged in evaluation)

---

### **2. Draw Prediction Difficulty**

**Issue**: Only 18% accuracy on draws

**Why**:
- Inherently random outcome
- Small sample effects
- Late goals, red cards, referee decisions

**Current Strategy**: Exclude draws from betting

**Impact**: Miss potential value bets on draws

**Solution**: Accept limitation (draws too noisy)

**Severity**: 🟡 Medium (acceptable trade-off)

---

### **3. Overfitting Still Present**

**Current**: 2.13% accuracy gap (Train 51.03% → Test 48.98%)

**Acceptable But Not Perfect**:
- Significant improvement
- But still room for improvement

**Why Still Present**:
- Limited data relative to noise
- Football inherently unpredictable
- Complex patterns hard to generalize

**Solution**: Continuous monitoring, more regularization if needed

**Severity**: 🟢 Low (2% gap acceptable)

---

### **4. Calibration Only Marginally Improves ROI**

**Current Results**:
```
Before Calibration: -4.51% ROI (on all bets)
After Calibration:  -3.45% ROI (on all bets)
Only +1.06% improvement
```

**Why**:
- Bookmakers already very well-calibrated
- Diminishing returns from calibration
- Need better strategy (selective betting) more than calibration

**Solution**: Focus on selective betting thresholds

**Severity**: 🟢 Low (calibration still useful for probabilities)

---

### **5. No Multi-Market Predictions**

**Current**: Only predict match outcome (1X2)

**Missing**:
- Over/Under goals
- Both Teams To Score (BTTS)
- Correct Score
- Handicap betting

**Impact**: Limited betting opportunities

**Solution**: Extend model to predict goals (regression), then derive other markets

**Severity**: 🟡 Medium (future enhancement)

---

## 🔧 Technical Limitations

### **1. No Real-Time Pipeline**

**Current**: Manual execution of scripts

**Missing**:
- Automated daily scraping
- Real-time predictions
- Automated bet detection
- Notifications (Telegram, email)

**Impact**: Can't use for live betting

**Solution**: Build production pipeline (Airflow + Docker)

**Severity**: 🔴 High (for production use)

---

### **2. No Model Monitoring**

**Current**: Train once, no continuous monitoring

**Missing**:
- Performance tracking over time
- Data drift detection
- Model degradation alerts
- Automatic retraining triggers

**Impact**: Model may degrade without notice

**Solution**: Implement MLOps monitoring (MLflow, Evidently)

**Severity**: 🟡 Medium

---

### **3. No A/B Testing Framework**

**Current**: Single model version

**Missing**:
- Multiple model comparison
- Gradual rollout
- Safe experimentation

**Impact**: Can't safely test improvements

**Solution**: Implement A/B testing infrastructure

**Severity**: 🟢 Low (not critical yet)

---

### **4. Limited Scalability**

**Current**: Single machine, local execution

**Issues**:
- Training takes ~30 minutes
- Scraping takes hours
- No parallel processing

**Solution**: Migrate to cloud (AWS/GCP) with distributed computing

**Severity**: 🟢 Low (current scale manageable)

---

## 📈 Evaluation Limitations

### **1. Short Test Period**

**Current**: 1.79 years of test data (4,628 matches)

**Issue**: 
- Relatively short for long-term validation
- Might not capture all conditions (e.g., COVID impact)

**Solution**: Continue accumulating data

**Severity**: 🟡 Medium

---

### **2. No Live Betting Validation**

**Current**: Only backtested on historical data

**Missing**:
- Real-world betting performance
- Impact of odds movement
- Market liquidity constraints
- Bet acceptance/rejection

**Impact**: Real performance may differ from backtest

**Solution**: Paper trading before real money

**Severity**: 🔴 High (for real deployment)

---

### **3. Limited Odds Coverage**

**Current Source**: football-data.co.uk
- Only major bookmakers (Bet365, Pinnacle, etc.)
- Not all matches have odds
- Some matches missing

**Impact**: 
- 4,628 test matches with odds
- 766 test matches without odds (excluded)

**Solution**: Scrape more bookmakers/Scrape oddsportal

**Severity**: 🟡 Medium

---

## 💰 Betting Limitations (less important)

### **1. No Real Money Testing**

**Status**: Only theoretical backtesting

**Missing**:
- Real bet execution
- Stake sizing strategies
- Bankroll management
- Psychological factors

**Impact**: Real ROI may differ

**Solution**: Start with small stakes for validation

**Severity**: 🔴 High (essential before real betting)

---

### **2. No Consideration of Betting Fees**

**Current**: Assume 0% fees/commissions

**Reality**:
- Bookmakers may charge fees
- Exchange betting has commissions (2-5%)
- Withdrawal fees

**Impact**: Real ROI will be lower

**Solution**: Factor in fees when calculating value bets

**Severity**: 🟡 Medium

---

### **3. No Market Liquidity Check**

**Issue**: Don't verify if odds are actually available

**Reality**:
- High odds may have low liquidity
- Odds change quickly
- Bet limits exist

**Impact**: Can't always place desired bets

**Solution**: Integrate live odds API to check availability

**Severity**: 🔴 High (for production)

---

### **4. No Stake Sizing Strategy**

**Current**: Assume fixed stake (€10 per bet)

**Missing**:
- Kelly Criterion
- Risk management
- Bankroll optimization

**Impact**: Suboptimal profit/risk balance

**Solution**: Implement dynamic stake sizing

**Severity**: 🟡 Medium

---

## 🔐 Legal & Ethical Limitations

### **1. Scraping Terms of Service**

**Issue**: Scraping may violate website ToS

**Risk**:
- IP bans
- Legal action (unlikely but possible)
- Data access revocation

**Mitigation**:
- Respect rate limits
- Use delays between requests
- For educational purposes only

**Severity**: 🟡 Medium (risk accepted)

---

### **2. Gambling Regulations**

**Issue**: Betting laws vary by country

**Constraints**:
- Some countries prohibit online betting
- Age restrictions
- Licensing requirements

**Solution**: User responsibility to comply with local laws

**Severity**: 🔴 High (legal compliance essential)

---

### **3. No Responsible Gambling Features**

**Missing**:
- Loss limits
- Addiction warnings
- Self-exclusion options

**Impact**: Potential harm to users

**Solution**: Do not bet and do not use this

**Severity**: 🟡 Medium (ethical responsibility)

---

## 🎯 Acknowledged Trade-offs

### **1. Accuracy vs Interpretability**

**Choice**: XGBoost (complex) over Linear Models (interpretable)

**Trade-off**:
- ✅ Better predictions
- ❌ Less interpretable

**Accepted**: Performance > interpretability for betting

---

### **2. Data Size vs Data Quality**

**Choice**: Clean data (36,940 matches) over all data (42,911)

**Trade-off**:
- ✅ Higher quality
- ❌ Less quantity

**Accepted**: Quality > quantity

---

### **3. Generalization vs Overfitting**

**Choice**: Strong regularization

**Trade-off**:
- ✅ Better test performance (48.98%)
- ❌ Worse train performance (51.03%)

**Accepted**: Generalization > train accuracy

---

## 📊 Known Issues

### **1. Brier Score Not Optimal**

**Current**: 0.6089 (test)
**Bookmaker**: 0.5993 (test)
**Gap**: +1.6%

**Why**: Bookmakers have more data and market dynamics

**Solution**: Accept limitation, focus on selective betting

---

### **2. ROI Negative on Global Betting**

**Current**: -3.45% ROI when betting on all matches

**Why**: Bookmakers better overall

**Solution**: Only bet selectively (Pure ROI: +16.62%)

---

### **3. Variance in Small Sample**

**Issue**: 24 bets/year (Pure ROI strategy) = high variance

**Risk**: Lucky/unlucky streaks

**Solution**: Use Volume strategy (198 bets/year) for diversification

---

## 🚀 Improvement Priorities

**High Priority** (Production Blockers):
1. 🔴 Scrape future odds (OddsPortal)
2. 🔴 Real-time prediction pipeline
3. 🔴 Automated notifications
4. 🔴 Transfermarkt player data

**Medium Priority** (Enhancement):
1. 🟡 Better calibration (Temperature Scaling)
2. 🟡 Multi-market predictions
3. 🟡 Model monitoring

**Low Priority** (Nice to Have):
1. 🟢 Weather data integration
2. 🟢 Expand leagues coverage
3. 🟢 Cloud deployment
4. 🟢 Dashboard UI

---

## ⚠️ Important Disclaimer

**This system is for educational and research purposes only.**

- ❌ Not financial advice
- ❌ No guarantees of profit
- ❌ Past performance ≠ future results
- ⚠️ Gambling involves risk of loss
- ⚠️ It can become an addiction

**Use responsibly and at your own risk, but Overall do not gamble.**

---

*Last Updated: February 2026*