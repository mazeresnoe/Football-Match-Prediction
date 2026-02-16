"""
Baseline Logistic Regression

Régression logistique multiclasse avec features simples :
- Elo
- Forme récente (5/10 derniers matchs)
- Buts marqués/encaissés
- H2H
- Avantage repos
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple
import joblib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as configs


class LogisticRegressionBaseline:
    """
    Régression logistique multiclasse (3 classes : Home, Draw, Away)
    """
    
    def __init__(self, features: list = None):
        """
        Args:
            features: Liste des features à utiliser
                     Si None, utilise LOGREG_FEATURES_NO_XG par défaut
        """
        self.name = "Logistic Regression"
        self.features = features or configs.LOGREG_FEATURES_NO_XG
        
        # Modèle
        self.model = LogisticRegression(
            multi_class='multinomial',   # Classification multiclasse
            solver='lbfgs',               # Solver adapté au multiclasse
            max_iter=1000,                # Augmenter si warning de convergence
            random_state=configs.RANDOM_STATE,
            C=1.0                         # Régularisation (plus petit = plus régularisé)
        )
        
        # Preprocessing
        self.scaler = StandardScaler()           # Normalisation des features
        self.imputer = SimpleImputer(strategy='median')  # Gestion valeurs manquantes
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Entraîne le modèle
        
        Args:
            df: DataFrame avec features + colonne 'result'
        """
        print(f"\n Entraînement {self.name}...")
        
        # Préparer X et y
        X, y = self._prepare_data(df)
        
        if len(X) == 0:
            raise ValueError("❌ Aucune donnée disponible pour l'entraînement")
        
        # Imputer les valeurs manquantes
        X_clean = self.imputer.fit_transform(X)
        
        # Normaliser (important pour LogReg)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Convertir y en indices (0=Home, 1=Draw, 2=Away)
        y_idx = y.map({1: 0, 0: 1, -1: 2})
        
        # Entraîner
        self.model.fit(X_scaled, y_idx)
        self.is_fitted = True
        
        # Stats
        print(f"    Modèle entraîné sur {len(X):,} matchs")
        print(f"    Features utilisées : {len(self.features)}")
        
        # Feature importance (coefficients)
        self._print_feature_importance()
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités
        
        Args:
            df: DataFrame avec les features
        
        Returns:
            Array (n, 3) avec [prob_home, prob_draw, prob_away]
        """
        if not self.is_fitted:
            raise ValueError("❌ Le modèle n'est pas entraîné. Appelle d'abord .fit()")
        
        # Préparer X
        X, _ = self._prepare_data(df)
        
        # Appliquer le même preprocessing que lors du fit
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        
        # Prédire les probabilités
        probs = self.model.predict_proba(X_scaled)
        
        return probs
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les features X et la target y
        
        Returns:
            (X, y) où X = DataFrame des features, y = Series des résultats
        """
        # Vérifier quelles features sont disponibles
        available_features = [f for f in self.features if f in df.columns]
        
        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"   ⚠️  {len(missing)} features manquantes : {list(missing)[:5]}...")
        
        if len(available_features) == 0:
            raise ValueError("❌ Aucune feature disponible dans le DataFrame")
        
        # Extraire X et y
        X = df[available_features].copy()
        y = df[configs.TARGET_COL] if configs.TARGET_COL in df.columns else None
        
        return X, y
    
    def _print_feature_importance(self):
        """
        Affiche les features les plus importantes (basé sur les coefficients)
        """
        # Coefficients pour chaque classe
        coefs = self.model.coef_  # Shape: (3, n_features) pour 3 classes
        
        # Moyenne absolue des coefficients sur les 3 classes
        importance = np.abs(coefs).mean(axis=0)
        
        # Trier
        feature_importance = pd.DataFrame({
            'feature': self.features[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n    Top 10 features importantes :")
        for i, row in feature_importance.head(10).iterrows():
            print(f"      {i+1:2}. {row['feature']:<30} : {row['importance']:.4f}")
    
    def save(self, filename: str):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            filename: Nom du fichier (ex: 'logreg_no_xg.pkl')
        """
        if not self.is_fitted:
            print("⚠️  Le modèle n'est pas entraîné, sauvegarde annulée")
            return
        
        filepath = configs.MODELS_DIR / filename
        
        # Sauvegarder tout (modèle + preprocessing)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'features': self.features,
            'is_fitted': self.is_fitted
        }, filepath)
        
        print(f"   ✓ Modèle sauvegardé : {filepath}")
    
    @classmethod
    def load(cls, filename: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filename: Nom du fichier
        
        Returns:
            Instance de LogisticRegressionBaseline
        """
        filepath = configs.MODELS_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
        
        # Charger
        data = joblib.load(filepath)
        
        # Recréer l'instance
        baseline = cls(features=data['features'])
        baseline.model = data['model']
        baseline.scaler = data['scaler']
        baseline.imputer = data['imputer']
        baseline.is_fitted = data['is_fitted']
        
        print(f" Modèle chargé : {filepath}")
        
        return baseline
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"LogisticRegressionBaseline(name='{self.name}', features={len(self.features)}, {status})"


if __name__ == "__main__":
    # Test rapide
    print("Test Baseline Logistic Regression\n")
    
    # Créer des données de test
    np.random.seed(42)
    n = 100
    
    test_data = pd.DataFrame({
        'elo_home': np.random.randint(1500, 1900, n),
        'elo_away': np.random.randint(1500, 1900, n),
        'elo_diff': np.random.randint(-300, 300, n),
        'elo_diff_squared': np.random.randint(0, 90000, n),
        'home_form_5': np.random.uniform(0, 3, n),
        'away_form_5': np.random.uniform(0, 3, n),
        'diff_form_5': np.random.uniform(-2, 2, n),
        'home_goals_scored_avg_5': np.random.uniform(0.5, 3, n),
        'away_goals_scored_avg_5': np.random.uniform(0.5, 3, n),
        'home_goals_conceded_avg_5': np.random.uniform(0.5, 2.5, n),
        'away_goals_conceded_avg_5': np.random.uniform(0.5, 2.5, n),
        'home_goal_diff_avg_5': np.random.uniform(-1, 2, n),
        'away_goal_diff_avg_5': np.random.uniform(-1, 2, n),
        'result': np.random.choice([1, 0, -1], n)
    })
    
    # Créer et entraîner le modèle
    features = ['elo_home', 'elo_away', 'elo_diff', 'home_form_5', 'away_form_5']
    model = LogisticRegressionBaseline(features=features)
    model.fit(test_data)
    
    # Prédire
    probs = model.predict_proba(test_data[:5])
    
    print(f"\n Prédictions (5 premiers matchs) :")
    for i in range(5):
        pred_class = ['Home', 'Draw', 'Away'][np.argmax(probs[i])]
        actual = {1: 'Home', 0: 'Draw', -1: 'Away'}[test_data.iloc[i]['result']]
        correct = "✓" if pred_class == actual else "✗"
        print(f"   Match {i+1}: {probs[i]} → {pred_class:5} (réel: {actual:5}) {correct}")