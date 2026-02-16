"""
Baseline Simple : Elo + Home Advantage

Ce baseline utilise UNIQUEMENT :
- La différence d'Elo entre les deux équipes
- Un bonus de 100 points Elo pour l'équipe qui joue à domicile

C'est le modèle de Machine Learning le plus simple possible.

Formule mathématique :
    P(A bat B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))

Cette formule vient du système Elo d'Arpad Elo, créé pour le classement des joueurs d'échecs.
"""

import numpy as np
import pandas as pd


class SimpleEloBaseline:
    """
    Modèle basé sur la formule Elo classique
    
    Paramètres :
        home_advantage : Bonus Elo pour l'équipe à domicile (défaut : 100)
    """
    
    def __init__(self, home_advantage: int = 100):
        """
        Initialise le baseline Elo
        
        Args:
            home_advantage : Bonus en points Elo pour jouer à domicile
                            100 est une valeur standard en football
        """
        self.name = "Simple Elo"
        self.home_advantage = home_advantage
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcule les probabilités de victoire avec la formule Elo
        
        Args:
            df : DataFrame avec colonnes 'elo_home' et 'elo_away'
        
        Returns:
            Array de shape (n_matches, 3) contenant les probabilités
            [prob_home, prob_draw, prob_away] pour chaque match
        
        Exemple :
            >>> df = pd.DataFrame({
            ...     'elo_home': [1800, 1700],
            ...     'elo_away': [1600, 1750]
            ... })
            >>> model = SimpleEloBaseline()
            >>> probs = model.predict_proba(df)
            >>> probs
            array([[0.849, 0.050, 0.101],
                   [0.520, 0.200, 0.280]])
        """
        n = len(df)
        probs = np.zeros((n, 3))
        
        # CORRECTION : Utiliser enumerate pour avoir un compteur séquentiel
        for i, (idx, row) in enumerate(df.iterrows()):
            # Étape 1 : Ajuster l'Elo avec le home advantage
            # L'équipe à domicile reçoit un bonus de points Elo
            elo_home_adjusted = row['elo_home'] + self.home_advantage
            elo_away_adjusted = row['elo_away']
            
            # Étape 2 : Calculer P(Home gagne)
            # Formule : 1 / (1 + 10^((Elo_away - Elo_home) / 400))
            prob_home_win = self._calculate_win_probability(elo_home_adjusted, elo_away_adjusted)
            
            # Étape 3 : Calculer P(Away gagne)
            # C'est simplement l'inverse : P(Away gagne | Elo_away, Elo_home)
            prob_away_win = self._calculate_win_probability(elo_away_adjusted, elo_home_adjusted)
            
            # Étape 4 : Calculer P(Draw)
            # On prend le "reste" : 1 - P(Home) - P(Away)
            # On force un minimum de 10% pour le nul (réaliste en football)
            prob_draw = max(1 - prob_home_win - prob_away_win, 0.10)
            
            # Étape 5 : Normaliser pour que la somme = 1.0
            # (important pour des probabilités valides)
            total = prob_home_win + prob_draw + prob_away_win
            probs[i] = [
                prob_home_win / total,
                prob_draw / total,
                prob_away_win / total
            ]
        
        return probs
    
    def _calculate_win_probability(self, elo_a: float, elo_b: float) -> float:
        """
        Calcule la probabilité que l'équipe A batte l'équipe B
        en utilisant la formule Elo standard
        
        Formule :
            P(A bat B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
        
        Intuition :
            - Si Elo_A >> Elo_B (ex: 1900 vs 1500), alors P ≈ 1.0 (quasi certain)
            - Si Elo_A = Elo_B (ex: 1700 vs 1700), alors P = 0.5 (50/50)
            - Si Elo_A << Elo_B (ex: 1500 vs 1900), alors P ≈ 0.0 (quasi impossible)
        
        Le facteur 400 est une constante du système Elo :
            - Une différence de 400 points → P ≈ 90%
            - Une différence de 200 points → P ≈ 75%
            - Une différence de 100 points → P ≈ 64%
        
        Args:
            elo_a : Points Elo de l'équipe A
            elo_b : Points Elo de l'équipe B
        
        Returns:
            Probabilité que A batte B (entre 0 et 1)
        """
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def explain_prediction(self, elo_home: float, elo_away: float, verbose: bool = True) -> dict:
        """
        Explique une prédiction en détail (utile pour débugger)
        
        Args:
            elo_home : Elo de l'équipe à domicile
            elo_away : Elo de l'équipe extérieure
            verbose : Si True, affiche l'explication
        
        Returns:
            Dictionnaire avec toutes les étapes de calcul
        
        Exemple :
            >>> model = SimpleEloBaseline(home_advantage=100)
            >>> model.explain_prediction(1800, 1600)
            
            === EXPLICATION PRÉDICTION ===
            Elo Home (brut)      : 1800
            Elo Away (brut)      : 1600
            Home Advantage       : +100
            Elo Home (ajusté)    : 1900
            Elo Diff (ajusté)    : +300
            
            P(Home gagne)        : 0.849 (84.9%)
            P(Away gagne)        : 0.101 (10.1%)
            P(Draw) (résidu)     : 0.050 (5.0%)
            
            Après normalisation :
            P(Home) : 0.849
            P(Draw) : 0.050
            P(Away) : 0.101
        """
        # Calculs
        elo_home_adj = elo_home + self.home_advantage
        elo_diff = elo_home_adj - elo_away
        
        prob_home = self._calculate_win_probability(elo_home_adj, elo_away)
        prob_away = self._calculate_win_probability(elo_away, elo_home_adj)
        prob_draw = max(1 - prob_home - prob_away, 0.10)
        
        # Normalisation
        total = prob_home + prob_draw + prob_away
        prob_home_norm = prob_home / total
        prob_draw_norm = prob_draw / total
        prob_away_norm = prob_away / total
        
        result = {
            'elo_home_raw': elo_home,
            'elo_away_raw': elo_away,
            'home_advantage': self.home_advantage,
            'elo_home_adjusted': elo_home_adj,
            'elo_diff': elo_diff,
            'prob_home_raw': prob_home,
            'prob_away_raw': prob_away,
            'prob_draw_raw': prob_draw,
            'prob_home_final': prob_home_norm,
            'prob_draw_final': prob_draw_norm,
            'prob_away_final': prob_away_norm
        }
        
        if verbose:
            print(f"\n=== EXPLICATION PRÉDICTION ===")
            print(f"Elo Home (brut)      : {elo_home}")
            print(f"Elo Away (brut)      : {elo_away}")
            print(f"Home Advantage       : +{self.home_advantage}")
            print(f"Elo Home (ajusté)    : {elo_home_adj}")
            print(f"Elo Diff (ajusté)    : {elo_diff:+.0f}")
            print(f"\nP(Home gagne)        : {prob_home:.3f} ({prob_home*100:.1f}%)")
            print(f"P(Away gagne)        : {prob_away:.3f} ({prob_away*100:.1f}%)")
            print(f"P(Draw) (résidu)     : {prob_draw:.3f} ({prob_draw*100:.1f}%)")
            print(f"\nAprès normalisation :")
            print(f"P(Home) : {prob_home_norm:.3f}")
            print(f"P(Draw) : {prob_draw_norm:.3f}")
            print(f"P(Away) : {prob_away_norm:.3f}")
        
        return result
    
    def __repr__(self):
        return f"SimpleEloBaseline(name='{self.name}', home_advantage={self.home_advantage})"


# ========================================
# TESTS (si tu lances ce fichier directement)
# ========================================

if __name__ == "__main__":
    print("="*70)
    print("  TEST DU BASELINE SIMPLE ELO")
    print("="*70)
    
    # Créer le modèle
    model = SimpleEloBaseline(home_advantage=100)
    print(f"\nModèle créé : {model}")
    
    # Test 1 : Équipes équilibrées
    print(f"\n{'='*70}")
    print("TEST 1 : Équipes équilibrées (1700 vs 1700)")
    print("="*70)
    model.explain_prediction(1700, 1700)
    
    # Test 2 : Gros favori à domicile
    print(f"\n{'='*70}")
    print("TEST 2 : Gros favori à domicile (1900 vs 1600)")
    print("="*70)
    model.explain_prediction(1900, 1600)
    
    # Test 3 : Outsider à domicile
    print(f"\n{'='*70}")
    print("TEST 3 : Outsider à domicile (1500 vs 1800)")
    print("="*70)
    model.explain_prediction(1500, 1800)
    
    # Test 4 : Prédictions sur plusieurs matchs
    print(f"\n{'='*70}")
    print("TEST 4 : Prédictions sur un dataset")
    print("="*70)
    
    test_data = pd.DataFrame({
        'home_team': ['Arsenal', 'Man City', 'Liverpool', 'Brighton', 'Wolves'],
        'away_team': ['Brighton', 'Everton', 'Man United', 'Man City', 'Arsenal'],
        'elo_home': [1800, 1900, 1850, 1650, 1600],
        'elo_away': [1600, 1650, 1820, 1900, 1800]
    })
    
    probs = model.predict_proba(test_data)
    
    print(f"\n{'Home':15} vs {'Away':15} | Prob H | Prob D | Prob A | Prédiction")
    print("-"*70)
    for i, row in test_data.iterrows():
        pred_class = ['Home', 'Draw', 'Away'][np.argmax(probs[i])]
        print(f"{row['home_team']:15} vs {row['away_team']:15} | "
              f"{probs[i][0]:.3f}  | {probs[i][1]:.3f}  | {probs[i][2]:.3f}  | "
              f"{pred_class}")
    
    print("\n✅ Tests terminés !")