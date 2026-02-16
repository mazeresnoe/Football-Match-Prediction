# XGBoost Football Prediction Model

Professional XGBoost implementation for football match prediction with value betting strategies.

## Quick Start

```bash
# Step 2a: Train baseline model
python core/step2a_baseline.py

# Step 2b: Optimize hyperparameters (50 trials, ~30 min)
python core/step2b_optimization.py

# Step 2c: Calibrate probabilities
python core/step2c_calibration.py

# Step 3: Find value bets
python core/step3_strategies.py
```

## Current Performance

- ROI: +3.12% (Conservative strategy)
- Win Rate: 53.3%
- Total Bets: 405 (on test set)
- Log Loss: 1.014
- Brier Score: 0.607

## Project Structure

```
xgboost/
├── core/              # Main training scripts (Step 2a, 2b, 2c, 3)
├── utils/             # Analysis and visualization utilities
├── configs/           # Feature configurations
├── docs/              # Detailed documentation
├── scripts/           # Utility scripts (diagnostics, etc.)
└── archive/           # Obsolete scripts
```

## Best Model

**Location**: `models/saved/production/xgboost_calibrated_no_xg.pkl`

- Features: 137
- Method: Isotonic calibration
- Strategy: Conservative (EV>10%, Confidence>55%)
- Training date: 2026-01-28

## Documentation

- [Installation Guide](docs/installation_guide.md)
- [Quick Start](docs/quick_start.md)
- [Complete Roadmap](docs/roadmap_complete.md)
- [Step 2b Details](docs/readme_step2b.md)

## Next Steps

1. Scrape OddsPortal for live odds
2. Create prediction pipeline for upcoming matches
3. Automate daily predictions
4. Track real-world performance

## Requirements

See `requirements.txt` in project root.

## License

Private project.