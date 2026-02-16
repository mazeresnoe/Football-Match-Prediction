# src\preprocessing\etape2\dual_dataset_splitter_v2.py
"""
Split robuste du dataset :
- NO XG  : toutes les lignes, colonnes xG supprimées
- WITH XG: uniquement les lignes avec has_xg_stats >= 0.5
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import numpy as np


class DualDatasetSplitterV2:
    def __init__(self, df: pd.DataFrame, xg_threshold: float = 0.1):
        """
        :param df: dataset complet
        :param xg_threshold: seuil minimal pour inclure un match dans WITH XG
        """
        self.df = df.copy()
        self.xg_threshold = xg_threshold
        self.log = []

    def split(self):
        print(" Démarrage du split V2...\n")

        df_no_xg, df_with_xg = self.create_dual_datasets()
        self.save(df_no_xg, df_with_xg)
        self.save_metadata(df_no_xg, df_with_xg)

        return df_no_xg, df_with_xg

    def create_dual_datasets(self):
        print(" Recherche dynamique des colonnes xG...")

        # Détection large des colonnes xG / expected goals
        xg_cols = [c for c in self.df.columns if any(k in c.lower() for k in ["xg", "expectedgoals"])]
        if not xg_cols:
            print("⚠️ AUCUNE colonne xG trouvée !")
            return self.df.copy(), self.df.iloc[0:0].copy()
        print(f"✅ {len(xg_cols)} colonnes xG trouvées")

        # ---------- NO XG ----------
        df_no_xg = self.df.copy()
        df_no_xg.drop(columns=xg_cols, inplace=True, errors="ignore")
        df_no_xg.sort_values("date", inplace=True)
        df_no_xg.reset_index(drop=True, inplace=True)

        # ---------- WITH XG ----------
        df_with_xg = self.df.copy()

        # Si la colonne 'has_xg_stats' existe, utiliser le seuil
        if "has_xg_stats" in df_with_xg.columns:
            mask_has_xg = df_with_xg["has_xg_stats"] >= self.xg_threshold
        else:
            # sinon fallback : au moins une valeur xG non NaN
            mask_has_xg = df_with_xg[xg_cols].notna().any(axis=1)

        df_with_xg = df_with_xg[mask_has_xg].copy()
        df_with_xg.sort_values("date", inplace=True)
        df_with_xg.reset_index(drop=True, inplace=True)

        # Log
        self.log.append({
            "n_total_rows": len(self.df),
            "n_xg_cols": len(xg_cols),
            "n_with_xg": int(mask_has_xg.sum()),
            "n_no_xg": len(df_no_xg),
            "xg_columns": xg_cols[:20]
        })

        print(f"✅ NO XG  : {len(df_no_xg):,} lignes")
        print(f"✅ WITH XG: {len(df_with_xg):,} lignes\n")

        return df_no_xg, df_with_xg

    def save(self, df_no_xg, df_with_xg):
        out = Path("data/clean/prematch/etape2/split")
        out.mkdir(parents=True, exist_ok=True)

        no_xg_path = out / "full_dataset_no_xg.csv"
        with_xg_path = out / "full_dataset_with_xg.csv"

        df_no_xg.to_csv(no_xg_path, index=False)
        df_with_xg.to_csv(with_xg_path, index=False)

        print(" Fichiers sauvegardés")

    def save_metadata(self, df_no_xg, df_with_xg):
        meta = {
            "created_at": datetime.now().isoformat(),
            "rows": {
                "total": len(self.df),
                "no_xg": len(df_no_xg),
                "with_xg": len(df_with_xg)
            },
            "xg_columns_detected": self.log[0]["xg_columns"],
            "xg_threshold_used": self.xg_threshold
        }

        meta_path = Path("data/clean/prematch/etape2/split/dataset_split_metadata_v2.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(" Metadata sauvegardée")


if __name__ == "__main__":
    input_file = "data/clean/prematch/etape2/full_dataset_v2.csv"
    df = pd.read_csv(input_file, parse_dates=['date'])

    splitter = DualDatasetSplitterV2(df)
    df_no_xg, df_with_xg = splitter.split()

    print("\n✅ Split V2 terminé !")




