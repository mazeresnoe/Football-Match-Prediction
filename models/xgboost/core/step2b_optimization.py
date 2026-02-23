"""
Step 2b OPTIMAL : Réduction Overfitting + Amélioration Modèle

STRATÉGIE:
1. Feature Selection (Minimal/Medium/Full)
2. TimeSeriesSplit pour optimisation (respecte temporalité)
3. Optimiser sur Brier Score (meilleure métrique pour probabilities)
4. Ensemble Multi-Seed (5 modèles, réduit variance)
5. Calibration Isotonic sur Train+CV (plus de data)

OBJECTIFS:
- Minimiser overfitting (train-test gap)
- Améliorer Brier Score (probabilities calibrées)
- Améliorer Log Loss (qualité prédictions)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths

from models.xgboost.configs.features_config import (
    XGBOOST_FEATURES_MINIMAL,
    XGBOOST_FEATURES_MEDIUM,
    XGBOOST_FEATURES_NO_XG_V1,
    XGBOOST_FEATURES_WITH_XG_V1,
)

# Import calibration depuis utils
from models.xgboost.utils.calibration import ManualCalibratedClassifier, ManualCalibratedEnsemble

sns.set_style("whitegrid")



# ========================================
# CLASSE XGBOOST AMÉLIORÉE
# ========================================

class XGBoostImproved:
    def __init__(self, features: list, params: dict = None, name: str = "XGBoost"):
        self.name = name
        self.features = features
        self.params = params or cfg.XGBOOST_PARAMS.copy()
        self.model = None
        self.is_fitted = False
        self.best_iteration = None
        
    def fit(self, train_df: pd.DataFrame, eval_set: list = None, 
            early_stopping_rounds: int = 50, verbose: bool = False):
        """Entraîne le modèle avec early stopping"""
        X_train = train_df[self.features].copy()
        y_train = train_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).to_numpy()
        
        self.model = xgb.XGBClassifier(**self.params)
        
        if eval_set:
            try:
                self.model = xgb.XGBClassifier(
                    **self.params,
                    early_stopping_rounds=early_stopping_rounds
                )
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
            except (TypeError, ValueError):
                print(f"   ⚠️ Early stopping non supporté")
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(X_train, y_train, verbose=verbose)
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
        
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            
        self.is_fitted = True
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("❌ Modèle non entraîné")
        X = df[self.features].copy()
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("❌ Modèle non entraîné")
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.features, 
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df


# ========================================
# ÉVALUATION COMPLÈTE
# ========================================

def evaluate_model_complete(model, train_df, cv_df, test_df, model_name: str = None):
    """Évalue un modèle sur TRAIN, CV et TEST"""
    results = []
    
    for dataset_name, df in [("train", train_df), ("cv", cv_df), ("test", test_df)]:
        y_true = df[cfg.TARGET_COL].values
        
        # Prédictions
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(df)
        else:
            probs = model.predict_proba(df)
        
        y_pred = np.argmax(probs, axis=1)
        y_pred_mapped = np.array([{0: 1, 1: 0, 2: -1}[p] for p in y_pred])
        y_true_idx = np.array([{1: 0, 0: 1, -1: 2}[y] for y in y_true])
        
        # Métriques
        acc = accuracy_score(y_true, y_pred_mapped)
        logloss = log_loss(y_true_idx, probs)
        
        # Brier Score
        y_true_onehot = np.zeros((len(y_true), 3))
        y_true_onehot[np.arange(len(y_true)), y_true_idx] = 1
        brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))
        
        # ROI
        roi = None
        profit = None
        n_bets = None
        
        if all(col in df.columns for col in ['odds_home', 'odds_draw', 'odds_away']):
            odds_df = df[['odds_home', 'odds_draw', 'odds_away']].copy()
            mask_odds = odds_df.notna().all(axis=1)
            
            if mask_odds.sum() > 0:
                eval_res = utils.evaluate_predictions(
                    y_true[mask_odds], 
                    probs[mask_odds], 
                    odds_df[mask_odds],
                    model_name or "Model"
                )
                roi = eval_res['roi']
                profit = eval_res['profit']
                n_bets = eval_res['n_bets']
        
        results.append({
            'model': model_name or "Model",
            'dataset': dataset_name,
            'accuracy': acc,
            'log_loss': logloss,
            'brier_score': brier,
            'roi': roi,
            'profit': profit,
            'n_bets': n_bets,
        })
    
    return pd.DataFrame(results)


# ========================================
# PHASE 1 : FEATURE SELECTION
# ========================================

def compare_feature_sets(train_df, cv_df, test_df, with_xg: bool = False):
    """Compare Minimal, Medium, Full features"""
    print(f"\n{'='*70}\n  COMPARAISON DES SETS DE FEATURES\n{'='*70}")
    
    feature_configs = {
        "Minimal (12 features)": XGBOOST_FEATURES_MINIMAL,
        "Medium (45 features)": XGBOOST_FEATURES_MEDIUM,
    }
    
    if with_xg:
        feature_configs["Full (140+ features)"] = XGBOOST_FEATURES_WITH_XG_V1
    else:
        feature_configs["Full (137 features)"] = XGBOOST_FEATURES_NO_XG_V1
    
    all_results = []
    
    for config_name, features in feature_configs.items():
        print(f"\n🔹 Test : {config_name}")
        
        # Filter available features
        available_features = [f for f in features if f in train_df.columns]
        missing = len(features) - len(available_features)
        if missing > 0:
            print(f"   ⚠️ {missing} features manquantes")
        print(f"   → {len(available_features)} features utilisées")
        
        # Create and train model
        model = XGBoostImproved(
            features=available_features,
            params=cfg.XGBOOST_PARAMS,
            name=config_name
        )
        
        # Prepare eval_set
        X_cv = cv_df[available_features]
        y_cv = cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2})
        
        model.fit(
            train_df, 
            eval_set=[(X_cv, y_cv)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        results = evaluate_model_complete(model, train_df, cv_df, test_df, config_name)
        all_results.append(results)
        
        # Print CV results
        cv_row = results[results['dataset'] == 'cv'].iloc[0]
        print(f"    CV → LogLoss: {cv_row['log_loss']:.4f}, Brier: {cv_row['brier_score']:.4f}")
    
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save
    result_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison.csv',
        with_xg=with_xg
    )
    final_results.to_csv(result_path, index=False)
    print(f"\n✅ Résultats : {result_path}")
    
    # Print comparison
    print(f"\n{'='*70}\n  TEST SET COMPARISON\n{'='*70}")
    test_results = final_results[final_results['dataset'] == 'test'].sort_values('log_loss')
    print(test_results[['model', 'log_loss', 'brier_score', 'accuracy']].to_string(index=False))
    
    # Return best config name
    best_config = test_results.iloc[0]['model']
    print(f"\n🏆 Meilleure configuration : {best_config}")
    
    return final_results, best_config


# ========================================
# PHASE 2 : HYPERPARAMETER OPTIMIZATION (Brier + TimeSeriesSplit)
# ========================================

def optimize_hyperparameters_optimal(train_df, features: list, n_trials: int = 100, 
                                    n_splits: int = 5, with_xg: bool = False):
    """
    Optimise les hyperparamètres avec:
    - TimeSeriesSplit (respecte temporalité)
    - Brier Score (meilleure métrique pour probabilities)
    - Regularization forte (prévient overfit)
    """
    print(f"\n{'='*70}")
    print(f"  OPTIMISATION HYPERPARAMÈTRES (Brier + TimeSeriesSplit)")
    print(f"{'='*70}")
    print(f"📊 Trials : {n_trials}")
    print(f"📊 Features : {len(features)}")
    print(f"📊 TimeSeriesSplit : {n_splits} folds")
    
    X_train = train_df[features].values
    y_train = train_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'gamma': trial.suggest_float('gamma', 0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'random_state': cfg.RANDOM_STATE,
            'n_jobs': -1,
        }
        
        brier_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            probs = model.predict_proba(X_val)
            
            # Brier Score
            y_val_onehot = np.zeros((len(y_val), 3))
            y_val_onehot[np.arange(len(y_val)), y_val] = 1
            brier = np.mean(np.sum((probs - y_val_onehot) ** 2, axis=1))
            brier_scores.append(brier)
        
        return np.mean(brier_scores)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=cfg.RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Optimisation terminée !")
    print(f"    Meilleur Brier : {study.best_value:.4f}")
    print(f"    Meilleurs paramètres :")
    for key, value in study.best_params.items():
        print(f"      • {key:20s} : {value}")
    
    # Sauvegarder
    params_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename=f'best_params_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
        with_xg=with_xg
    )
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"    Paramètres : {params_path}")
    
    # Graphique
    try:
        history_path = SavePaths.get_result_path(
            category='step2b_optimization',
            filename='optuna_history.html',
            with_xg=with_xg
        )
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(history_path))
        print(f"    Historique : {history_path}")
    except:
        pass
    
    return study.best_params


# ========================================
# PHASE 3 : ENTRAÎNEMENT ENSEMBLE MULTI-SEED + CALIBRATION
# ========================================

def train_final_ensemble(train_df, cv_df, features, final_params, n_models=5):
    """
    Entraîne un ensemble de modèles avec seeds différentes + calibration.
    
    POURQUOI:
    - Réduit variance (différentes initialisations)
    - Améliore robustesse
    - Calibration sur Train+CV (plus de data)
    """
    print(f"\n{'='*70}")
    print(f"  ENTRAÎNEMENT ENSEMBLE ({n_models} modèles)")
    print(f"{'='*70}")
    
    models = []
    
    X_cv = cv_df[features].values
    y_cv = cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).values
    
    for i, seed in enumerate(range(cfg.RANDOM_STATE, cfg.RANDOM_STATE + n_models)):
        print(f"🔹 Modèle {i+1}/{n_models} (seed={seed})...")
        
        params_seed = final_params.copy()
        params_seed['random_state'] = seed
        
        model = XGBoostImproved(
            features=features, 
            params=params_seed, 
            name=f"XGBoost_seed_{seed}"
        )
        
        model.fit(
            train_df, 
            eval_set=[(cv_df[features], y_cv)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        models.append(model)
    
    print(f"✅ {n_models} modèles entraînés")
    
    # Calibration sur Train+CV
    print(f"\n🔧 Calibration Isotonic (Train+CV)...")
    
    train_cv_df = pd.concat([train_df, cv_df], ignore_index=True)
    X_train_cv = train_cv_df[features].values
    y_train_cv = train_cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).values
    
    ensemble = ManualCalibratedEnsemble(models)
    ensemble.fit(X_train_cv, y_train_cv)
    
    print(f"✅ Calibration terminée")
    
    return ensemble, models


# ========================================
# FONCTION PRINCIPALE
# ========================================

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     STEP 2B OPTIMAL : RÉDUCTION OVERFITTING + AMÉLIORATION   ║
╚══════════════════════════════════════════════════════════════╝

STRATÉGIE:
1. Feature Selection (Minimal/Medium/Full)
2. TimeSeriesSplit (respecte temporalité)
3. Optimisation sur Brier Score (probabilities calibrées)
4. Ensemble Multi-Seed (réduit variance)
5. Calibration Isotonic sur Train+CV (plus de data)
    """)
    
    xgb_version = xgb.__version__
    print(f"XGBoost version : {xgb_version}\n")
    
    with_xg = False
    
    # 1. Charger données
    print("📂 Chargement des données...")
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    
    print(f"✅ Données chargées")
    print(f"   Train : {len(train_df)}, CV : {len(cv_df)}, Test : {len(test_df)}\n")
    
    # ========================================
    # PHASE 1 : FEATURE SELECTION
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 1 : FEATURE SELECTION")
    print(f"{'#'*70}")
    
    results_features, best_config = compare_feature_sets(train_df, cv_df, test_df, with_xg=with_xg)
    
    # Sélectionner les features
    if "Full" in best_config:
        best_features = XGBOOST_FEATURES_WITH_XG_V1 if with_xg else XGBOOST_FEATURES_NO_XG_V1
    elif "Medium" in best_config:
        best_features = XGBOOST_FEATURES_MEDIUM
    else:
        best_features = XGBOOST_FEATURES_MINIMAL
    
    best_features = [f for f in best_features if f in train_df.columns]
    print(f"\n🎯 Features sélectionnées : {len(best_features)}")
    
    # ========================================
    # PHASE 2 : HYPERPARAMETER OPTIMIZATION
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 2 : HYPERPARAMETER OPTIMIZATION")
    print(f"{'#'*70}")
    
    best_params = optimize_hyperparameters_optimal(
        train_df,
        features=best_features,
        n_trials=100,
        n_splits=5,
        with_xg=with_xg
    )
    
    # ========================================
    # PHASE 3 : ENTRAÎNEMENT ENSEMBLE
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 3 : ENTRAÎNEMENT ENSEMBLE + CALIBRATION")
    print(f"{'#'*70}")
    
    final_params = cfg.XGBOOST_PARAMS.copy()
    final_params.update(best_params)
    
    ensemble, models_list = train_final_ensemble(
        train_df, cv_df,
        features=best_features,
        final_params=final_params,
        n_models=5
    )
    
    # ========================================
    # ÉVALUATION FINALE
    # ========================================
    
    print(f"\n{'='*70}")
    print(f"  ÉVALUATION FINALE")
    print(f"{'='*70}")
    
    final_results = evaluate_model_complete(
        ensemble, train_df, cv_df, test_df,
        "XGBoost Optimized Ensemble"
    )
    
    print(f"\n{final_results.to_string(index=False)}")
    
    # Calcul overfitting gap
    train_row = final_results[final_results['dataset']=='train'].iloc[0]
    test_row = final_results[final_results['dataset']=='test'].iloc[0]
    
    acc_gap = (train_row['accuracy'] - test_row['accuracy']) * 100
    logloss_gap = ((test_row['log_loss'] - train_row['log_loss']) / train_row['log_loss']) * 100
    brier_gap = ((test_row['brier_score'] - train_row['brier_score']) / train_row['brier_score']) * 100
    
    print(f"\n📊 OVERFITTING ANALYSIS")
    print(f"   • Accuracy gap : {acc_gap:.2f}%")
    print(f"   • Log Loss gap : {logloss_gap:+.2f}%")
    print(f"   • Brier gap    : {brier_gap:+.2f}%")
    
    if acc_gap < 3 and logloss_gap < 6 and brier_gap < 6:
        print(f"\n✅ EXCELLENT : Overfitting contrôlé !")
    elif acc_gap < 5 and logloss_gap < 10:
        print(f"\n✅ BON : Overfitting acceptable")
    else:
        print(f"\n⚠️ ATTENTION : Overfitting encore présent")
    
    # Feature importance
    print(f"\n📊 Top 20 Features :")
    
    # Moyenne sur tous les modèles
    all_importances = []
    for m in models_list:
        imp = m.get_feature_importance(top_n=len(best_features))
        all_importances.append(imp)
    
    # Merge et moyenne
    feat_imp_df = all_importances[0][['feature']].copy()
    for i, imp_df in enumerate(all_importances):
        feat_imp_df = feat_imp_df.merge(
            imp_df.rename(columns={'importance': f'imp_{i}'}),
            on='feature'
        )
    
    feat_imp_df['importance_mean'] = feat_imp_df.iloc[:, 1:].mean(axis=1)
    feat_imp_df = feat_imp_df.sort_values('importance_mean', ascending=False)
    
    print(feat_imp_df[['feature', 'importance_mean']].head(20).to_string(index=False))
    
    # Sauvegarder
    import joblib
    
    model_path = SavePaths.get_model_path(
        category='experiments',
        model_name='xgboost_optimized',
        with_xg=with_xg
    )
    
    joblib.dump({
        'ensemble': ensemble,
        'models': models_list,
        'features': best_features,
        'params': final_params,
    }, model_path)
    
    print(f"\n✅ Modèle sauvegardé : {model_path}")
    
    # Résultats
    result_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='final_results.csv',
        with_xg=with_xg
    )
    final_results.to_csv(result_path, index=False)
    
    SavePaths.save_metadata(
        category='step2b_optimization',
        filename='final_results.csv',
        metadata={
            'n_trials': 100,
            'n_models': 5,
            'n_features': len(best_features),
            'best_config': best_config,
            'test_brier': float(test_row['brier_score']),
            'test_logloss': float(test_row['log_loss']),
            'overfitting_gap': float(acc_gap),
            'optimization_metric': 'brier_score',
            'cv_method': 'TimeSeriesSplit',
            'calibration': 'isotonic'
        },
        with_xg=with_xg
    )
    
    print(f"✅ Résultats sauvegardés : {result_path}")
    
    print(f"\n{'='*70}")
    print(f"  TERMINÉ ✅")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()