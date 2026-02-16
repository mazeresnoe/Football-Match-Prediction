"""
Step 2b : Amélioration du XGBoost Baseline
- Feature selection progressive
- Hyperparameter tuning avec Optuna
- Comparaison des versions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from datetime import datetime

# Imports locaux
sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as cfg
import models.utils as utils
from models.configs.save_paths import SavePaths

# Import de la config des features
from models.xgboost.configs.features_config import (
    XGBOOST_FEATURES_MINIMAL,
    XGBOOST_FEATURES_MEDIUM,
    XGBOOST_FEATURES_NO_XG_V1,
    XGBOOST_FEATURES_WITH_XG_V1,
)

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
        
        # Créer le modèle
        self.model = xgb.XGBClassifier(**self.params)
        
        # Stratégie robuste pour l'early stopping
        if eval_set:
            # Méthode 1 : Essayer avec early_stopping_rounds dans le constructeur (XGBoost 2.0+)
            try:
                self.model = xgb.XGBClassifier(
                    **self.params,
                    early_stopping_rounds=early_stopping_rounds
                )
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
                
            except (TypeError, ValueError):
                # Méthode 2 : Essayer sans early stopping (fallback)
                print(f"   ⚠️ Early stopping non supporté, entraînement sans early stopping")
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(X_train, y_train, verbose=verbose)
        else:
            # Pas d'eval set, entraînement normal
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
    """Évalue un modèle sur TRAIN, CV et TEST avec toutes les métriques"""
    results = []
    
    for dataset_name, df in [("train", train_df), ("cv", cv_df), ("test", test_df)]:
        y_true = df[cfg.TARGET_COL].values
        probs = model.predict_proba(df)
        
        # Prédictions classiques
        y_pred = np.argmax(probs, axis=1)
        y_pred_mapped = np.array([{0: 1, 1: 0, 2: -1}[p] for p in y_pred])
        
        # Convertir y_true pour log_loss (doit être en indices 0, 1, 2)
        y_true_idx = np.array([{1: 0, 0: 1, -1: 2}[y] for y in y_true])
        
        # Métriques de base
        acc = accuracy_score(y_true, y_pred_mapped)
        logloss = log_loss(y_true_idx, probs)
        
        # Brier score (version robuste)
        # Mapping : 1 (Home) → 0, 0 (Draw) → 1, -1 (Away) → 2
        mapping = {1: 0, 0: 1, -1: 2}
        y_true_idx_check = np.array([mapping[y] for y in y_true])

        # Vérifier que y_true_idx est bien calculé
        assert np.array_equal(y_true_idx, y_true_idx_check), "Bug dans le mapping y_true → y_true_idx"

        # One-hot encoding
        y_true_onehot = np.zeros((len(y_true), 3))
        y_true_onehot[np.arange(len(y_true)), y_true_idx] = 1

        # Calcul du Brier
        brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))

        # Debug (optionnel, à supprimer après)
        if dataset_name == 'test':
            print(f"\n🔍 Brier Score Debug:")
            print(f"   Min/Max/Mean probs : [{probs.min():.3f}, {probs.max():.3f}, {probs.mean():.3f}]")
            print(f"   Brier : {brier:.4f}")
            print(f"   Expected range : [0.15, 0.25]")
        
        # ROI si odds disponibles
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
                    model_name or model.name
                )
                roi = eval_res['roi']
                profit = eval_res['profit']
                n_bets = eval_res['n_bets']
        
        results.append({
            'model': model_name or model.name,
            'dataset': dataset_name,
            'accuracy': acc,
            'log_loss': logloss,
            'brier_score': brier,
            'roi': roi,
            'profit': profit,
            'n_bets': n_bets,
            'n_features': len(model.features),
        })
    
    return pd.DataFrame(results)


# ========================================
# FEATURE SELECTION PROGRESSIVE
# ========================================

def compare_feature_sets(train_df, cv_df, test_df, with_xg: bool = False):
    """Compare différentes configurations de features"""
    print(f"\n{'='*70}\n  COMPARAISON DES SETS DE FEATURES\n{'='*70}")
    
    feature_configs = {
        "Minimal (12 features)": XGBOOST_FEATURES_MINIMAL,
        "Medium (50 features)": XGBOOST_FEATURES_MEDIUM,
    }
    
    if with_xg:
        feature_configs["V1 Full XG (140+ features)"] = XGBOOST_FEATURES_WITH_XG_V1
    else:
        feature_configs["V1 Full (130+ features)"] = XGBOOST_FEATURES_NO_XG_V1
    
    all_results = []
    
    for config_name, features in feature_configs.items():
        print(f"\n Test : {config_name}")
        
        # Vérifier que toutes les features existent
        available_features = [f for f in features if f in train_df.columns]
        missing = set(features) - set(available_features)
        
        if missing:
            print(f"   ⚠️ Features manquantes : {len(missing)} ({', '.join(list(missing)[:5])}...)")
            features = available_features
        
        print(f"   → {len(features)} features utilisées")
        
        # Créer et entraîner le modèle
        model = XGBoostImproved(
            features=features,
            params=cfg.XGBOOST_PARAMS,
            name=config_name
        )
        
        # Préparer eval_set
        X_cv = cv_df[features]
        y_cv = cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2})
        
        model.fit(
            train_df, 
            eval_set=[(X_cv, y_cv)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Évaluer
        results = evaluate_model_complete(model, train_df, cv_df, test_df, config_name)
        all_results.append(results)
        
        # Afficher résultats CV
        cv_row = results[results['dataset'] == 'cv'].iloc[0]
        print(f"    CV → Acc: {cv_row['accuracy']:.4f}, LogLoss: {cv_row['log_loss']:.4f}, " + 
              f"Brier: {cv_row['brier_score']:.4f}, ROI: {cv_row['roi']:.2f}%")
    
    # Compiler tous les résultats
    final_results = pd.concat(all_results, ignore_index=True)
    return final_results


# ========================================
# HYPERPARAMETER TUNING AVEC OPTUNA
# ========================================

def optimize_hyperparameters(train_df, cv_df, features: list, n_trials: int = 50, 
                            with_xg: bool = False):
    """Optimise les hyperparamètres avec Optuna"""
    print(f"\n{'='*70}\n  OPTIMISATION DES HYPERPARAMÈTRES (Optuna)\n{'='*70}")
    print(f" {n_trials} essais avec {len(features)} features")
    
    # Préparer les données
    X_train = train_df[features].values
    y_train = train_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).values
    X_cv = cv_df[features].values
    y_cv = cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2}).values
    
    def objective(trial):
        # Définir l'espace de recherche
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'random_state': cfg.RANDOM_STATE,
            'eval_metric': 'mlogloss',
            'n_jobs': -1,
        }
        
        # Entraîner avec early stopping robuste
        try:
            # Méthode 1 : early_stopping_rounds dans le constructeur (XGBoost 2.0+)
            params_with_es = params.copy()
            params_with_es['early_stopping_rounds'] = 30
            model = xgb.XGBClassifier(**params_with_es)
            model.fit(X_train, y_train, eval_set=[(X_cv, y_cv)], verbose=False)
            
        except (TypeError, ValueError):
            # Méthode 2 : Sans early stopping (fallback)
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
        
        # Prédire et évaluer
        probs = model.predict_proba(X_cv)
        logloss = log_loss(y_cv, probs)
        
        return logloss
    
    # Créer et lancer l'étude
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=cfg.RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Optimisation terminée !")
    print(f"    Meilleur LogLoss : {study.best_value:.4f}")
    print(f"    Meilleurs paramètres :")
    for key, value in study.best_params.items():
        print(f"      • {key:20s} : {value}")
    
    # Sauvegarder les résultats
    results_dir = cfg.RESULTS_WITH_XG_DIR if with_xg else cfg.RESULTS_NO_XG_DIR
    
    # Sauvegarder les meilleurs paramètres
    params_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename=f'best_params_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
        with_xg=with_xg
    )
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"    Paramètres sauvegardés : {params_path}")
    
    # Créer un graphique d'optimisation
    try:
        history_path = SavePaths.get_result_path(
            category='step2b_optimization',
            filename='optuna_history.html',
            with_xg=with_xg
        )
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(history_path))
        print(f"    Historique Optuna : {history_path}")
    except:
        pass
    
    return study.best_params


# ========================================
# FONCTION PRINCIPALE
# ========================================

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          STEP 2B : AMÉLIORATION XGBOOST BASELINE              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Vérification de la version XGBoost
    xgb_version = xgb.__version__
    print(f"  XGBoost version : {xgb_version}")
    major, minor = map(int, xgb_version.split('.')[:2])
    if major >= 2:
        print(f"   → Utilisation des callbacks pour early stopping")
    else:
        print(f"   → Utilisation du parameter pour early stopping")
    print()
    
    # Choix : avec ou sans XG
    with_xg = False  # 🔹 Change ici pour tester avec XG
    
    # 1. Charger les données
    df = utils.load_data(with_xg=with_xg, merge_odds=True)
    train_df, cv_df, test_df = utils.train_cv_test_split(df)
    
    # ========================================
    # PHASE 1 : COMPARAISON DES FEATURE SETS
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 1 : COMPARAISON DES CONFIGURATIONS DE FEATURES")
    print(f"{'#'*70}")
    
    results_features = compare_feature_sets(train_df, cv_df, test_df, with_xg=with_xg)
    
    # Sauvegarder
    result_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='feature_comparison.csv',
        with_xg=with_xg
    )
    results_features.to_csv(result_path, index=False)
    print(f"\n✅ Résultats sauvegardés : {result_path}")
    
    # Afficher le tableau comparatif
    print(f"\n{'='*70}\n  RÉSULTATS COMPARATIFS (TEST SET)\n{'='*70}")
    test_results = results_features[results_features['dataset'] == 'test'].copy()
    test_results = test_results.sort_values('log_loss')
    print(test_results[['model', 'accuracy', 'log_loss', 'brier_score', 'roi']].to_string(index=False))
    
    # Identifier les meilleures features
    best_config = test_results.iloc[0]['model']
    print(f"\n Meilleure configuration : {best_config}")
    
    # ========================================
    # PHASE 2 : HYPERPARAMETER TUNING
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 2 : OPTIMISATION DES HYPERPARAMÈTRES")
    print(f"{'#'*70}")
    
    # Utiliser les features de la meilleure config
    if "Full" in best_config:
        best_features = XGBOOST_FEATURES_WITH_XG_V1 if with_xg else XGBOOST_FEATURES_NO_XG_V1
    elif "Medium" in best_config:
        best_features = XGBOOST_FEATURES_MEDIUM
    else:
        best_features = XGBOOST_FEATURES_MINIMAL
    
    # Filtrer les features disponibles
    best_features = [f for f in best_features if f in train_df.columns]
    
    print(f"   Optimisation avec {len(best_features)} features")
    
    # Lancer l'optimisation
    best_params = optimize_hyperparameters(
        train_df, cv_df, 
        features=best_features,
        n_trials=50,  # 🔹 Augmente pour un meilleur résultat (ex: 100-200)
        with_xg=with_xg
    )
    
    # ========================================
    # PHASE 3 : ENTRAÎNER LE MODÈLE FINAL
    # ========================================
    
    print(f"\n{'#'*70}")
    print("  PHASE 3 : ENTRAÎNEMENT DU MODÈLE OPTIMISÉ")
    print(f"{'#'*70}")
    
    # Mettre à jour les paramètres
    final_params = cfg.XGBOOST_PARAMS.copy()
    final_params.update(best_params)
    
    # Créer le modèle final
    final_model = XGBoostImproved(
        features=best_features,
        params=final_params,
        name="XGBoost Optimized"
    )
    
    # Entraîner
    X_cv = cv_df[best_features]
    y_cv = cv_df[cfg.TARGET_COL].map({1: 0, 0: 1, -1: 2})
    
    final_model.fit(
        train_df,
        eval_set=[(X_cv, y_cv)],
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Évaluer
    final_results = evaluate_model_complete(
        final_model, train_df, cv_df, test_df,
        "XGBoost Optimized"
    )
    
    # Afficher les résultats finaux
    print(f"\n{'='*70}\n  RÉSULTATS FINAUX\n{'='*70}")
    print(final_results.to_string(index=False))
    
    # Feature importance
    print(f"\n Top 20 Features importantes :")
    print(final_model.get_feature_importance(top_n=20).to_string(index=False))
    
    # Sauvegarder le modèle
    import joblib
    model_path = SavePaths.get_model_path(
        category='experiments',
        model_name='xgboost_optimized',
        with_xg=with_xg
    )
    joblib.dump({
        'model': final_model.model,
        'features': final_model.features,
        'params': final_params,
        'best_iteration': final_model.best_iteration,
    }, model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}")
    
    # Sauvegarder tous les résultats
    result_path = SavePaths.get_result_path(
        category='step2b_optimization',
        filename='final_results.csv',
        with_xg=with_xg
    )
    final_results.to_csv(result_path, index=False)

    # Sauvegarder les métadonnées
    SavePaths.save_metadata(
        category='step2b_optimization',
        filename='final_results.csv',
        metadata={
            'n_trials': 50,
            'best_log_loss': final_results[final_results['dataset']=='test']['log_loss'].min(),
            'best_roi': final_results[final_results['dataset']=='test']['roi'].max(),
            'n_features': len(final_model.features)
        },
        with_xg=with_xg
    )
    print(f"✅ Résultats sauvegardés : {result_path}")
    

if __name__ == "__main__":
    main()