"""
Script de vérification de la compatibilité XGBoost
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

def check_xgboost_version():
    """Vérifie la version de XGBoost et la compatibilité"""
    
    print("="*70)
    print("  VÉRIFICATION DE LA CONFIGURATION XGBOOST")
    print("="*70)
    
    # 1. Vérifier que XGBoost est installé
    try:
        import xgboost as xgb
        print(f"\n✅ XGBoost installé")
        print(f"   Version : {xgb.__version__}")
    except ImportError:
        print("\n❌ XGBoost n'est pas installé !")
        print("   Installe-le avec : pip install xgboost")
        return False
    
    # 2. Vérifier la version
    version = xgb.__version__
    major, minor = map(int, version.split('.')[:2])
    
    if major >= 2:
        print(f"    Version moderne (>= 2.0)")
        print(f"      → Utilise les callbacks pour early stopping")
        
        # Vérifier que les callbacks sont disponibles
        try:
            from xgboost.callback import EarlyStopping
            print(f"   ✅ EarlyStopping callback disponible")
        except (ImportError, AttributeError):
            print(f"   ⚠️  Callbacks non disponibles, utilisation du fallback")
    else:
        print(f"    Version ancienne (< 2.0)")
        print(f"      → Utilise early_stopping_rounds parameter")
        print(f"      → Recommandation : mise à jour vers >= 2.0")
        print(f"      → Commande : pip install --upgrade xgboost")
    
    # 3. Test rapide d'entraînement
    print(f"\n Test rapide d'entraînement...")
    try:
        from sklearn.datasets import make_classification
        import numpy as np
        
        # Créer des données test
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3, 
                                   n_informative=3, random_state=42)
        
        # Split simple
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Test avec early_stopping_rounds dans le constructeur (XGBoost 2.0+)
        print(f"   → Test méthode 1 (early_stopping_rounds dans constructeur)...")
        try:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=50,
                early_stopping_rounds=10,
                random_state=42
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)
            print(f"   ✅ Méthode 1 fonctionne ! (shape: {preds.shape})")
            success = True
            
        except (TypeError, ValueError) as e:
            print(f"   ❌ Méthode 1 échoue : {type(e).__name__}")
            
            # Test sans early stopping (fallback)
            print(f"   → Test méthode 2 (sans early stopping)...")
            try:
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    n_estimators=10,
                    random_state=42
                )
                model.fit(X_train, y_train, verbose=False)
                preds = model.predict_proba(X_val)
                print(f"   ✅ Méthode 2 (fallback) fonctionne ! (shape: {preds.shape})")
                print(f"   ⚠️  Note : Early stopping non disponible, mais entraînement OK")
                success = True
                
            except Exception as e2:
                print(f"   ❌ Méthode 2 échoue aussi : {type(e2).__name__}: {e2}")
                success = False
        
        if success:
            print(f"\n✅ TOUT FONCTIONNE CORRECTEMENT !")
        else:
            print(f"\n❌ PROBLÈME DÉTECTÉ !")
            
        return success
        
    except Exception as e:
        print(f"\n❌ ERREUR lors du test : {e}")
        print(f"   Type d'erreur : {type(e).__name__}")
        return False
    
    finally:
        print(f"\n{'='*70}")


def check_dependencies():
    """Vérifie toutes les dépendances nécessaires"""
    
    print("\n" + "="*70)
    print("  VÉRIFICATION DES DÉPENDANCES")
    print("="*70)
    
    dependencies = {
        'xgboost': 'XGBoost',
        'optuna': 'Optuna (hyperparameter tuning)',
        'plotly': 'Plotly (visualisations interactives)',
        'matplotlib': 'Matplotlib (graphiques)',
        'seaborn': 'Seaborn (graphiques)',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
    }
    
    all_ok = True
    
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'N/A')
            print(f"✅ {display_name:40s} : {version}")
        except ImportError:
            print(f"❌ {display_name:40s} : NON INSTALLÉ")
            all_ok = False
    
    print("="*70)
    
    if all_ok:
        print("\n✅ Toutes les dépendances sont installées !")
    else:
        print("\n⚠️ Certaines dépendances manquent.")
        print("   Installe-les avec : pip install xgboost optuna plotly")
    
    return all_ok


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         DIAGNOSTIC ENVIRONNEMENT - XGBOOST IMPROVED           ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Vérifier les dépendances
    deps_ok = check_dependencies()
    
    # Vérifier XGBoost spécifiquement
    xgb_ok = check_xgboost_version()
    
    # Résumé final
    print("\n" + "="*70)
    print("  RÉSUMÉ")
    print("="*70)
    
    if deps_ok and xgb_ok:
        print("\n✅ TON ENVIRONNEMENT EST PRÊT !")
        print("   Tu peux lancer : python xgboost_improved.py")
    else:
        print("\n⚠️ Quelques problèmes à résoudre :")
        if not deps_ok:
            print("   • Installe les dépendances manquantes")
        if not xgb_ok:
            print("   • Vérifie ta version de XGBoost")
            print("   • Considère une mise à jour : pip install --upgrade xgboost")
    
    print("\n" + "="*70)