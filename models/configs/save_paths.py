"""
Système de gestion automatique des chemins de sauvegarde.

Ce module centralise tous les chemins de sauvegarde pour les modèles et résultats.
Il gère automatiquement la création des dossiers et le versioning.

Usage:
    from models.configs.save_paths import SavePaths
    
    # Sauvegarder un modèle
    model_path = SavePaths.get_model_path('production', 'xgboost_calibrated')
    joblib.dump(model, model_path)
    
    # Sauvegarder un résultat
    result_path = SavePaths.get_result_path('step3_strategies', 'strategies_results.csv')
    df.to_csv(result_path, index=False)
"""

from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
import json


class SavePaths:
    """
    Gestionnaire centralisé des chemins de sauvegarde.
    
    Gère automatiquement:
    - La création des dossiers
    - Le versioning des fichiers
    - L'organisation par date
    """
    
    # Chemins de base (depuis global_config)
    BASE_DIR = Path(__file__).parent.parent.parent.resolve()
    MODELS_SAVED = BASE_DIR / "models" / "saved"
    RESULTS_NO_XG = BASE_DIR / "results" / "modeling" / "no_xg"
    RESULTS_XG = BASE_DIR / "results" / "modeling" / "xg"
    
    # Structure des dossiers
    MODEL_CATEGORIES = ['production', 'experiments', 'archive']
    RESULT_CATEGORIES = [
        'production',
        'step1_baselines',
        'step2a_baseline', 
        'step2b_optimization',
        'step2c_calibration',
        'step3_strategies',
        'experiments',
        'archive'
    ]
    
    @classmethod
    def _ensure_dir(cls, path: Path) -> Path:
        """Crée le dossier s'il n'existe pas."""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def get_model_path(
        cls,
        category: Literal['production', 'experiments', 'archive'],
        model_name: str,
        version: Optional[str] = None,
        with_xg: bool = False
    ) -> Path:
        """
        Obtenir le chemin pour sauvegarder un modèle.
        
        Args:
            category: Catégorie du modèle (production/experiments/archive)
            model_name: Nom du modèle (ex: 'xgboost_calibrated')
            version: Version optionnelle (ex: 'v1.0', 'v1.1')
            with_xg: Si True, inclut '_xg' dans le nom
            
        Returns:
            Path complet pour la sauvegarde
            
        Example:
            >>> path = SavePaths.get_model_path('production', 'xgboost_calibrated', 'v1.0')
            >>> # models/saved/production/current/xgboost_calibrated_v1.0_no_xg.pkl
        """
        # Construire le nom du fichier
        xg_suffix = '_xg' if with_xg else '_no_xg'
        version_suffix = f'_{version}' if version else ''
        filename = f"{model_name}{version_suffix}{xg_suffix}.pkl"
        
        # Chemin selon la catégorie
        if category == 'production':
            # Production: current/ et history/
            current_dir = cls._ensure_dir(cls.MODELS_SAVED / 'production' / 'current')
            return current_dir / filename
            
        elif category == 'experiments':
            # Experiments: par date
            date_str = datetime.now().strftime('%Y-%m-%d')
            exp_dir = cls._ensure_dir(cls.MODELS_SAVED / 'experiments' / date_str)
            return exp_dir / filename
            
        else:  # archive
            archive_dir = cls._ensure_dir(cls.MODELS_SAVED / 'archive')
            return archive_dir / filename
    
    @classmethod
    def get_result_path(
        cls,
        category: str,
        filename: str,
        with_xg: bool = False,
        use_date_folder: bool = False
    ) -> Path:
        """
        Obtenir le chemin pour sauvegarder un résultat.
        
        Args:
            category: Catégorie du résultat (ex: 'step3_strategies')
            filename: Nom du fichier (ex: 'strategies_results.csv')
            with_xg: Si True, utilise le dossier 'xg' au lieu de 'no_xg'
            use_date_folder: Si True, crée un sous-dossier avec la date
            
        Returns:
            Path complet pour la sauvegarde
            
        Example:
            >>> path = SavePaths.get_result_path('step3_strategies', 'strategies_results.csv')
            >>> # results/modeling/no_xg/step3_strategies/strategies_results.csv
            
            >>> path = SavePaths.get_result_path('production', 'strategies.csv', use_date_folder=True)
            >>> # results/modeling/no_xg/production/2026-02-11/strategies.csv
        """
        # Choisir le dossier de base
        base_dir = cls.RESULTS_XG if with_xg else cls.RESULTS_NO_XG
        
        # Construire le chemin
        category_dir = base_dir / category
        
        if use_date_folder:
            date_str = datetime.now().strftime('%Y-%m-%d')
            result_dir = cls._ensure_dir(category_dir / date_str)
        else:
            result_dir = cls._ensure_dir(category_dir)
        
        return result_dir / filename
    
    @classmethod
    def archive_current_model(cls, model_name: str, with_xg: bool = False):
        """
        Archive le modèle actuel en production avant d'en sauvegarder un nouveau.
        
        Args:
            model_name: Nom du modèle (ex: 'xgboost_calibrated')
            with_xg: Si le modèle utilise xG
            
        Example:
            >>> SavePaths.archive_current_model('xgboost_calibrated')
            >>> # Déplace production/current/xgboost_calibrated_no_xg.pkl
            >>> # vers production/history/xgboost_calibrated_no_xg_2026-02-11.pkl
        """
        current_dir = cls.MODELS_SAVED / 'production' / 'current'
        history_dir = cls._ensure_dir(cls.MODELS_SAVED / 'production' / 'history')
        
        xg_suffix = '_xg' if with_xg else '_no_xg'
        current_file = current_dir / f"{model_name}{xg_suffix}.pkl"
        
        if current_file.exists():
            # Ajouter date au nom
            date_str = datetime.now().strftime('%Y-%m-%d')
            archive_file = history_dir / f"{model_name}{xg_suffix}_{date_str}.pkl"
            
            # Renommer si le fichier existe déjà (ajouter heure)
            if archive_file.exists():
                time_str = datetime.now().strftime('%H%M')
                archive_file = history_dir / f"{model_name}{xg_suffix}_{date_str}_{time_str}.pkl"
            
            current_file.rename(archive_file)
            print(f"Modèle actuel archivé: {archive_file.name}")
    
    @classmethod
    def clean_old_experiments(cls, days: int = 30):
        """
        Supprime les expériences de plus de X jours.
        
        Args:
            days: Nombre de jours à garder
            
        Example:
            >>> SavePaths.clean_old_experiments(30)
            >>> # Supprime tous les dossiers dans experiments/ plus vieux que 30 jours
        """
        experiments_dir = cls.MODELS_SAVED / 'experiments'
        if not experiments_dir.exists():
            return
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
        deleted_count = 0
        
        for date_folder in experiments_dir.iterdir():
            if date_folder.is_dir():
                # Vérifier la date de modification
                if date_folder.stat().st_mtime < cutoff_date:
                    # Supprimer le dossier et son contenu
                    import shutil
                    shutil.rmtree(date_folder)
                    deleted_count += 1
        
        if deleted_count > 0:
            print(f"Nettoyé {deleted_count} dossiers d'expériences obsolètes")
    
    @classmethod
    def save_metadata(cls, category: str, filename: str, metadata: dict, with_xg: bool = False):
        """
        Sauvegarde les métadonnées associées à un résultat.
        
        Args:
            category: Catégorie du résultat
            filename: Nom du fichier de résultat
            metadata: Dictionnaire de métadonnées
            with_xg: Si le résultat concerne les données avec xG
            
        Example:
            >>> metadata = {
            ...     'date': '2026-02-11',
            ...     'n_trials': 50,
            ...     'best_roi': 3.12
            ... }
            >>> SavePaths.save_metadata('step2b_optimization', 'results.csv', metadata)
        """
        base_dir = cls.RESULTS_XG if with_xg else cls.RESULTS_NO_XG
        category_dir = base_dir / category
        
        # Nom du fichier de métadonnées
        meta_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
        meta_path = category_dir / meta_filename
        
        # Ajouter timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        
        # Sauvegarder
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def get_latest_model(cls, category: str = 'production', with_xg: bool = False) -> Optional[Path]:
        """
        Obtenir le chemin du modèle le plus récent.
        
        Gère automatiquement les structures :
        - production : cherche dans production/current/
        - experiments : cherche dans experiments/YYYY-MM-DD/
        - archive : cherche dans archive/
        
        Args:
            category: Catégorie à chercher
            with_xg: Chercher dans les modèles avec xG
            
        Returns:
            Path du modèle le plus récent, ou None si aucun
        """
        if category == 'production':
            current_dir = cls.MODELS_SAVED / 'production' / 'current'
        else:
            current_dir = cls.MODELS_SAVED / category
        
        if not current_dir.exists():
            return None
        
        # Pattern de recherche
        xg_suffix = '_xg.pkl' if with_xg else '_no_xg.pkl'
        pkl_files = []
        
        # Pour experiments : chercher récursivement dans les sous-dossiers par date
        if category == 'experiments':
            for date_folder in current_dir.iterdir():
                if date_folder.is_dir():
                    for pkl_file in date_folder.glob('*.pkl'):
                        if pkl_file.name.endswith(xg_suffix):
                            pkl_files.append(pkl_file)
        else:
            # Pour production/archive : chercher directement
            pkl_files = [f for f in current_dir.glob('*.pkl') if f.name.endswith(xg_suffix)]
        
        if not pkl_files:
            return None
        
        # Retourner le plus récent (par date de modification)
        return max(pkl_files, key=lambda p: p.stat().st_mtime)


# Fonction helper pour l'import facile
def get_model_path(*args, **kwargs):
    """Raccourci pour SavePaths.get_model_path()"""
    return SavePaths.get_model_path(*args, **kwargs)


def get_result_path(*args, **kwargs):
    """Raccourci pour SavePaths.get_result_path()"""
    return SavePaths.get_result_path(*args, **kwargs)


if __name__ == "__main__":
    # Test du système
    print("Test du système de chemins de sauvegarde\n")
    
    # Test modèle production
    model_path = SavePaths.get_model_path('production', 'xgboost_calibrated', 'v1.0')
    print(f"Modèle production: {model_path}")
    
    # Test modèle experiment
    exp_path = SavePaths.get_model_path('experiments', 'xgboost_test')
    print(f"Modèle experiment: {exp_path}")
    
    # Test résultat
    result_path = SavePaths.get_result_path('step3_strategies', 'strategies_results.csv')
    print(f"Résultat: {result_path}")
    
    # Test résultat avec date
    dated_path = SavePaths.get_result_path('production', 'strategies.csv', use_date_folder=True)
    print(f"Résultat avec date: {dated_path}")