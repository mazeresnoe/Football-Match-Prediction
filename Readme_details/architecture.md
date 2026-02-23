Football-Match-Prediction/
│
├── .env                               # Environment variables (API keys, secrets)
├── .gitignore                         # Git ignore rules (data/, models/, etc.)
├── requirements.txt                   # Python dependencies
├── structure.txt                      # Project structure documentation
│
├── data/                              # All datasets (NOT versioned in Git)
│
│   ├── raw/                           # Raw scraped data (no modification)
│   │
│   │   ├── event_ids/                 # SofaScore event_id per league & season
│   │   │   ├── premier_league/        # English Premier League
│   │   │   ├── la_liga/               # Spanish La Liga
│   │   │   ├── bundesliga/            # German Bundesliga
│   │   │   ├── serie_a/               # Italian Serie A
│   │   │   ├── ligue_1/               # French Ligue 1
│   │   │   ├── championship/          # English Championship
│   │   │   ├── la_liga_2/             # Spanish Segunda División
│   │   │   ├── 2_bundesliga/          # German 2. Bundesliga
│   │   │   ├── serie_b/               # Italian Serie B
│   │   │   ├── ligue_2/               # French Ligue 2
│   │   │   ├── UEFA_champions_league/ # UEFA Champions League
│   │   │   ├── UEFA_europa_league/    # UEFA Europa League
│   │   │   ├── UEFA_conference_league/# UEFA Conference League
│   │   │   └── Fifa_world_cup_club/   # FIFA Club World Cup
│   │   │
│   │   ├── match_stats/               # Match-wise statistics (CSV + JSON)
│   │   │   └── (same league structure as event_ids)
│   │   │
│   │   ├── merged/                    # Raw merged dataset before cleaning
│   │   │   └── dataset_raw_merged.csv
│   │   │
│   │   └── scraping_progress/         # Scraping tracker (resume capability)
│   │
│   ├── clean/                         # Cleaned & ML-ready datasets
│   │
│   │   ├── post_match/                # Post-match enriched dataset (training only)
│   │   │   └── post_match_clean.csv
│   │   │
│   │   └── prematch/                  # ONLY features available before kickoff
│   │
│   │       ├── etape1/                # Initial merged dataset
│   │       │   └── full_dataset.csv
│   │       │
│   │       ├── etape2/                # Dataset split & structural improvements
│   │       │   ├── full_dataset_v2.csv
│   │       │   └── split/             # XG vs No-XG dataset variants
│   │       │       ├── full_dataset_no_xg.csv
│   │       │       ├── full_dataset_with_xg.csv
│   │       │       └── dataset_split_metadata_v2.json
│   │       │
│   │       └── etape3/                # Final cleaned training datasets
│   │           ├── full_dataset_no_xg_clean.csv
│   │           ├── full_dataset_no_xg_clean_v2.csv
│   │           ├── full_dataset_no_xg_clean_metadata.json
│   │           ├── full_dataset_with_xg_clean.csv
│   │           ├── full_dataset_with_xg_clean_v2.csv
│   │           └── full_dataset_with_xg_clean_metadata.json
│   │
│   ├── odds/                          # Betting odds datasets
│   │   ├── raw/                       # Raw bookmaker odds per league
│   │   │   ├── premier_league.csv
│   │   │   ├── la_liga.csv
│   │   │   ├── bundesliga.csv
│   │   │   ├── serie_a.csv
│   │   │   ├── ligue_1.csv
│   │   │   ├── championship.csv
│   │   │   ├── la_liga_2.csv
│   │   │   ├── bundesliga_2.csv
│   │   │   ├── serie_b.csv
│   │   │   └── ligue_2.csv
│   │   │
│   │   ├── all_odds_clean.csv         # Cleaned odds dataset
│   │   └── all_odds_standardized.csv  # Standardized odds format (model-ready)
│   │
│   └── eda/                           # Exploratory Data Analysis outputs
│       └── etape2/
│           ├── no_xg_dataset/         # EDA results (no xG version)
│           └── with_xg_dataset/       # EDA results (with xG version)
│
├── preprocessing/                     # Data cleaning & preparation scripts
│   ├── 1_etape/
│   │   ├── 1_raw_merger.py            # Merge raw league datasets
│   │   ├── 2_cleaning.py              # Initial cleaning & validation
│   │   └── eda/                       # Early-stage exploratory analysis
│   │
│   └── etape2/
│       ├── 5_dual_dataset_splitter_v2.py  # Split XG / No-XG datasets
│       ├── 7_cleaning.py                  # Advanced cleaning
│       ├── 8_feature_recovery.py          # Recover missing engineered features
│       └── 6_eda/                         # Intermediate EDA analysis
│
├── src/                                 # Core production code
│   ├── scraping/                        # Data collection layer
│   │   ├── sofascore_scrap/
│   │   │   ├── match_stats_scraper.py   # Extract match statistics via API
│   │   │   └── round_scrapper_2.py      # Round-based event scraping
│   │   │
│   │   └── odds/
│   │       ├── scrape_odds.py           # Odds scraper
│   │       └── team_mapping_complete.py # Team name harmonization
│   │
│   ├── feature_engineering/             # Feature construction logic
│   │   └── etape1/
│   │       ├── 3_feature_eng_v1_basic.py    # Basic rolling stats
│   │       └── 4_feature_eng_v2_advanced.py # Advanced derived metrics
│   │
│   └── utils/
│       └── mapping/
│           └── map.py                   # League/team mapping utilities
│
├── models/                              # ML modeling layer
│   ├── baseline/                        # Baseline models
│   │   ├── baseline_simple.py           # Simple statistical baseline
│   │   ├── baseline_logreg.py           # Logistic regression baseline
│   │   ├── baseline_bookmaker.py        # Bookmaker implied probability benchmark
│   │   ├── run_step1_benchmarks.py      # Run baseline experiments
│   │   └── test_baselines.py            # Baseline validation tests
│   │
│   ├── xgboost/                         # Main ML pipeline (XGBoost)
│   │   ├── core/
│   │   │   ├── step2a_baseline.py       # Initial XGB model
│   │   │   ├── step2b_optimization.py   # Hyperparameter tuning
│   │   │   ├── step2c_calibration.py    # Probability calibration
│   │   │   ├── step3_strategies.py      # Betting strategy comparison
│   │   │   └── value_bet_detector.py    # Value bet identification
│   │   │
│   │   ├── configs/
│   │   │   └── features_config.py       # Feature selection configuration
│   │   │
│   │   ├── scripts/
│   │   │   └── check_environment.py     # Environment validation script
│   │   │
│   │   └── utils/
│   │       ├── calibration.py
│   │       ├── error_analysis.py
│   │       ├── model_comparison.py
│   │       └── visualization.py
│   │
│   ├── configs/                         # Global configuration
│   │   ├── global_config.py
│   │   └── save_paths.py
│   │
│   ├── utils/                           # Modeling utilities
│   │   ├── data_utils.py
│   │   ├── eval_utils.py
│   │   └── other_utils.py
│   │
│   └── saved/                           # Saved trained models
│       ├── experiments/                 # Experimental runs
│       └── production/                  # Final selected models
│
├── results/                             # Model outputs & performance reports
│   └── modeling/
│       ├── no_xg/                       # Results for No-XG dataset
│       │   ├── production/
│       │   ├── step1_baselines/
│       │   ├── step2a_baseline/
│       │   ├── step2b_optimization/
│       │   └── step2c_calibration/
│       │
│       └── xg/                          # Results for XG-enhanced dataset