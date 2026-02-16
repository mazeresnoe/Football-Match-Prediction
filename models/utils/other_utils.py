"""
Fonctions pour sauvegarde / chargement de modèles
"""

import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as cfg

def save_model(model_obj, filename: str, models_dir: Path):
    filepath = models_dir / filename
    joblib.dump(model_obj, filepath)
    print(f" Modèle sauvegardé : {filepath}")


def load_model(filename: str, models_dir: Path):
    filepath = models_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Modèle introuvable : {filepath}")
    model_obj = joblib.load(filepath)
    print(f" Modèle chargé : {filepath}")
    return model_obj

def save_results(results_list: list, filename: str, with_xg: bool = False) -> Path:
    """
    Sauvegarde les résultats dans un fichier CSV

    Args:
        results_list: liste de dictionnaires (résultats)
        filename: nom du fichier CSV
        with_xg: si True → dossier XG, sinon dossier no_XG

    Returns:
        Path vers le fichier sauvegardé
    """
    df = pd.DataFrame(results_list)
    
    # Choisir le bon dossier
    results_dir = cfg.RESULTS_WITH_XG_DIR if with_xg else cfg.RESULTS_NO_XG_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"\n Résultats sauvegardés : {filepath}")
    return filepath