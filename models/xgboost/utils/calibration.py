"""
Classe de calibration manuelle pour XGBoost.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class ManualCalibratedClassifier:
    """
    Wrapper pour calibrer manuellement un modèle avec isotonic regression.
    """
    def __init__(self, base_model, method='isotonic'):
        self.base_model = base_model
        self.method = method
        self.calibrators = []
        self.is_fitted = False
        
    def fit(self, X, y):
        probs_raw = self.base_model.predict_proba(X)
        n_classes = probs_raw.shape[1]
        self.calibrators = []
        
        for i in range(n_classes):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            y_binary = (y == i).astype(float)
            prob_raw_class = probs_raw[:, i]
            calibrator.fit(prob_raw_class, y_binary)
            self.calibrators.append(calibrator)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Modèle non calibré")
        
        probs_raw = self.base_model.predict_proba(X)
        probs_calibrated = np.zeros_like(probs_raw)
        
        for i, calibrator in enumerate(self.calibrators):
            probs_calibrated[:, i] = calibrator.predict(probs_raw[:, i])
        
        row_sums = probs_calibrated.sum(axis=1, keepdims=True)
        probs_calibrated = probs_calibrated / row_sums
        
        return probs_calibrated
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class ManualCalibratedEnsemble:
    """Calibre un ensemble de modèles avec Isotonic Regression"""
    
    def __init__(self, models: list):
        self.models = models
        self.calibrators = [None, None, None]  # Un par classe
        self.is_calibrated = False
    
    def fit(self, X, y):
        """Calibre sur les probabilités moyennes de l'ensemble"""
        # Prédictions moyennes
        probs_list = [m.predict_proba(pd.DataFrame(X, columns=self.models[0].features)) 
                     for m in self.models]
        probs_avg = np.mean(probs_list, axis=0)
        
        # Calibrer chaque classe
        for class_idx in range(3):
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs_avg[:, class_idx], (y == class_idx).astype(int))
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
    
    def predict_proba(self, df):
        """Prédit avec calibration"""
        # Prédictions moyennes
        probs_list = [m.predict_proba(df) for m in self.models]
        probs_avg = np.mean(probs_list, axis=0)
        
        if not self.is_calibrated:
            return probs_avg
        
        # Appliquer calibration
        probs_calibrated = np.zeros_like(probs_avg)
        for class_idx in range(3):
            probs_calibrated[:, class_idx] = self.calibrators[class_idx].predict(
                probs_avg[:, class_idx]
            )
        
        # Renormaliser (somme = 1)
        probs_calibrated = probs_calibrated / probs_calibrated.sum(axis=1, keepdims=True)
        
        return probs_calibrated