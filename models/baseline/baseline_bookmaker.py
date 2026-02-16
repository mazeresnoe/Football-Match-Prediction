"""
Ce baseline est le VRAI benchmark à battre.

Baselines Bookmaker : Deux versions (Simple et Proportionnelle)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from models.utils.data_utils import odds_to_probs_simple, odds_to_probs_proportional


class BookmakerSimpleBaseline:
    """
    Baseline Bookmaker avec méthode SIMPLE (normalization uniforme).
    Moins précise mais plus simple.
    """
    
    def __init__(self):
        self.name = "Bookmaker (Simple)"
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités à partir des odds (méthode simple)
        """
        n = len(df)
        probs = np.zeros((n, 3))
        
        for i, row in enumerate(df.itertuples(index=False)):
            p_home, p_draw, p_away = odds_to_probs_simple(
                row.odds_home, 
                row.odds_draw, 
                row.odds_away
            )
            probs[i] = [p_home, p_draw, p_away]
        
        return probs
    
    def __repr__(self):
        return f"BookmakerSimpleBaseline(name='{self.name}')"


class BookmakerProportionalBaseline:
    """
    Baseline Bookmaker avec méthode PROPORTIONNELLE.
    Plus précise et recommandée (méthode des pros).
    """
    
    def __init__(self):
        self.name = "Bookmaker (Proportional)"
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités à partir des odds (méthode proportionnelle)
        """
        n = len(df)
        probs = np.zeros((n, 3))
        
        for i, row in enumerate(df.itertuples(index=False)):
            p_home, p_draw, p_away = odds_to_probs_proportional(
                row.odds_home, 
                row.odds_draw, 
                row.odds_away
            )
            probs[i] = [p_home, p_draw, p_away]
        
        return probs
    
    def __repr__(self):
        return f"BookmakerProportionalBaseline(name='{self.name}')"


# Alias pour rétro-compatibilité
BookmakerBaseline = BookmakerProportionalBaseline  # Par défaut = Proportionnelle


if __name__ == "__main__":
    # Test rapide
    print("Test Baselines Bookmaker\n")
    
    # Données de test
    test_data = pd.DataFrame({
        'odds_home': [6.80, 2.50, 1.50],
        'odds_draw': [4.99, 3.20, 4.00],
        'odds_away': [1.48, 2.80, 7.00]
    })
    
    # Tester les 2 méthodes
    models = [
        BookmakerSimpleBaseline(),
        BookmakerProportionalBaseline()
    ]
    
    for model in models:
        print(f"\n{'='*70}")
        print(f"  {model.name}")
        print(f"{'='*70}\n")
        
        probs = model.predict_proba(test_data)
        
        print(f"{'Cotes':<20} | {'Prob Home':>10} | {'Prob Draw':>10} | {'Prob Away':>10} | {'Total':>8}")
        print(f"{'-'*70}")
        
        for i, row in test_data.iterrows():
            cotes = f"{row['odds_home']:.2f} | {row['odds_draw']:.2f} | {row['odds_away']:.2f}"
            print(f"{cotes:<20} | {probs[i][0]:>10.3f} | {probs[i][1]:>10.3f} | {probs[i][2]:>10.3f} | {probs[i].sum():>8.4f}")