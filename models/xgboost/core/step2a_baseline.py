#  xgboost_baseline.py (finalisé version minimale nécessaire)

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))
import models.configs.global_config as configs
import models.utils as utils
from models.configs.save_paths import SavePaths


class XGBoostBaseline:
    def __init__(self, features: list = None, params: dict = None):
        self.name = "XGBoost Baseline"
        self.features = features or configs.LOGREG_FEATURES_NO_XG

        default_params = {
            'objective': 'multi:softprob',  # Probabilités pour 3 classes
            'num_class': 3,                 # Home, Draw, Away
            'n_estimators': 100,            # Nombre d'arbres
            'max_depth': 6,                 # Profondeur max
            'learning_rate': 0.1,           # Vitesse d'apprentissage
            'subsample': 0.8,               # % de données par arbre
            'colsample_bytree': 0.8,        # % de features par arbre
            'random_state': configs.RANDOM_STATE,
            'eval_metric': 'mlogloss',      # Métrique à optimiser
            'n_jobs': -1,                   # Parallélisation
        }

        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, eval_set: list = None, verbose: bool = True):
        """
        Entraîne le modèle XGBoost
        """
        if verbose:
            print(f"\n Entraînement {self.name}...")

        # Préparer X et y
        X = df[self.features].copy()
        y = df[configs.TARGET_COL].copy()

        # Convertir y en indices (0, 1, 2)
        y_idx = y.map({1: 0, 0: 1, -1: 2}).to_numpy()

        # Entraîner
        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(X, y_idx, **fit_params)
        self.is_fitted = True
        if verbose:
            print(f"   ✅ Entraînement terminé")


    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("❌ Le modèle n'est pas entraîné. Appelle d'abord .fit()")
        X = df[self.features].copy()
        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20, plot: bool = False) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("❌ Le modèle n'est pas entraîné")
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': self.features, 'importance': importances}) \
                        .sort_values('importance', ascending=False).head(top_n)
        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Features - XGBoost')
            plt.xlabel('Importance (Gain)')
            plt.tight_layout()
            plot_path = SavePaths.get_result_path(
                category='step2a_baseline',
                filename='feature_importance.png',
                with_xg=False
            )
            plt.savefig(plot_path, dpi=150)
            plt.show()
            print(f"   ✅ Graphique sauvegardé : {plot_path}")
        return importance_df

    def save(self, filename: str, category: str = 'experiments'):
        """
        Sauvegarde le modèle.
        
        Args:
            filename: Nom de base du fichier (sans extension)
            category: Catégorie de sauvegarde ('production', 'experiments', 'archive')
        """
        if not self.is_fitted:
            print("⚠️ Le modèle n'est pas entraîné")
            return
        
        model_path = SavePaths.get_model_path(
            category=category,
            model_name=filename,
            with_xg=False
        )
        joblib.dump({
            'model': self.model, 
            'features': self.features, 
            'is_fitted': self.is_fitted
        }, model_path)
        print(f"   ✅ Modèle sauvegardé : {model_path}")

    @classmethod
    def load(cls, filename: str = None, category: str = 'experiments'):
        """
        Charge un modèle sauvegardé.
        
        Args:
            filename: Nom de base du fichier (optionnel si on cherche le dernier)
            category: Catégorie où chercher
        """
        if filename:
            # Charger un fichier spécifique
            filepath = SavePaths.get_model_path(
                category=category,
                model_name=filename,
                with_xg=False
            )
        else:
            # Charger le plus récent
            filepath = SavePaths.get_latest_model(category, with_xg=False)
            
        if not filepath or not filepath.exists():
            raise FileNotFoundError(f"❌ Modèle introuvable : {filepath}")
        
        data = joblib.load(filepath)
        xgb_model = cls(features=data['features'])
        xgb_model.model = data['model']
        xgb_model.is_fitted = data['is_fitted']
        print(f"   ✅ Modèle chargé : {filepath}")
        return xgb_model


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              XGBOOST BASELINE - PREMIER TEST                 ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 1. Charger les données (with_xg=False / True si besoin)
    df = utils.load_data(with_xg=False, merge_odds=True)

    # 2. Split
    train_df, cv_df, test_df = utils.train_cv_test_split(df)

    # 3. Créer le modèle
    xgb_model = XGBoostBaseline()

    # 4. Eval set pour CV
    X_cv = cv_df[xgb_model.features]
    y_cv = cv_df[configs.TARGET_COL].map({1: 0, 0: 1, -1: 2})

    xgb_model.fit(train_df, eval_set=[(X_cv, y_cv)])

    # 5. Feature importance
    print(f"\n Top 15 features importantes :")
    print(xgb_model.get_feature_importance(top_n=15).to_string(index=False))

    # 6. Évaluation sur CV
    y_cv_true = cv_df[configs.TARGET_COL].values
    probs_cv = xgb_model.predict_proba(cv_df)
    odds_cv = cv_df[['odds_home', 'odds_draw', 'odds_away']].copy()
    mask_odds_cv = odds_cv.notna().all(axis=1)
    res_cv = utils.evaluate_predictions(
        y_cv_true[mask_odds_cv], probs_cv[mask_odds_cv], odds_cv[mask_odds_cv], "XGBoost Baseline"
    )
    utils.print_evaluation_summary(res_cv, "CV")

    # 7. Évaluation sur Test
    y_test_true = test_df[configs.TARGET_COL].values
    probs_test = xgb_model.predict_proba(test_df)
    odds_test = test_df[['odds_home', 'odds_draw', 'odds_away']].copy()
    mask_odds_test = odds_test.notna().all(axis=1)
    res_test = utils.evaluate_predictions(
        y_test_true[mask_odds_test], probs_test[mask_odds_test], odds_test[mask_odds_test], "XGBoost Baseline"
    )
    utils.print_evaluation_summary(res_test, "TEST")

    # 8. Comparer avec baselines
    baseline_path = SavePaths.get_result_path(
        category='step2a_baseline',
        filename='baseline_comparison_no_xg.csv',
        with_xg=False
    )

    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path)
        baselines_test = baselines[baselines['dataset'] == 'test']
    else:
        baselines_test = pd.DataFrame()

    xgb_results = pd.DataFrame([{
        'model': 'XGBoost Baseline',
        'dataset': 'test',
        'accuracy': res_test['accuracy'],
        'log_loss': res_test['log_loss'],
        'brier_score': res_test['brier_score'],
        'roi': res_test['roi']
    }])

    result_df = pd.concat([baselines_test, xgb_results], ignore_index=True)
    result_df.to_csv(baseline_path, index=False)
    print(f"✅ Comparaison baselines sauvegardée : {baseline_path}")


if __name__ == "__main__":
    main()
