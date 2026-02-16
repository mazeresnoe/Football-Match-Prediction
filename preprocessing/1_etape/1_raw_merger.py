# ============================================
# RAW MATCH STATS MERGER
# ============================================

import pandas as pd
from pathlib import Path
from typing import List
import warnings
import re
import numpy as np

warnings.filterwarnings("ignore")

# ============================================
# FONCTIONS D'UNIFORMISATION
# ============================================

def normalize_team_name(name: str) -> str:
    """ Nettoie et uniformise les noms d'équipes """
    if pd.isna(name):
        return name

    name = name.lower().strip()

    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "bayern munich": "bayern münchen",
        "psg": "paris saint-germain"
    }

    return replacements.get(name, name)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """ Uniformise les noms de colonnes """

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    column_map = {
        "home": "home_team",
        "away": "away_team",
        "hometeam": "home_team",
        "awayteam": "away_team"
    }

    df = df.rename(columns=column_map)

    return df

# ============================================
# AJOUT DE LA COLONNE RESULT
# ============================================

def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """ Ajoute la colonne 'result' après 'homescore' et 'awayscore' """
    if all(col in df.columns for col in ["homescore", "awayscore"]):
        conditions = [
            df["homescore"] > df["awayscore"],  # home win
            df["homescore"] < df["awayscore"],  # away win
            df["homescore"] == df["awayscore"]  # draw
        ]
        choices = [1, -1, 0]
        df["result"] = np.select(conditions, choices)

        # Réordonner pour placer result après awayscore
        cols = df.columns.tolist()
        cols.remove("result")
        idx = cols.index("awayscore") + 1
        cols.insert(idx, "result")
        df = df[cols]
    else:
        print("⚠️ Colonnes 'homescore' et/ou 'awayscore' manquantes, impossible de créer 'result'.")
    return df

# ============================================
# MERGER PRINCIPAL
# ============================================

class RawMatchStatsMerger:
    def __init__(self, base_path="data/raw/match_stats", verbose=True):
        self.base_path = Path(base_path)
        self.verbose = verbose

        # Dossier de sortie
        self.output_dir = Path("data/raw/merged")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # LOAD SAFE
    # -----------------------------------------
    def load_csv_safe(self, path: Path):
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            df["league"] = path.parent.name
            return df
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Erreur fichier: {path} -> {e}")
            return None

    # -----------------------------------------
    # LOAD TOUS LES FICHIERS
    # -----------------------------------------
    def load_all_files(self) -> List[pd.DataFrame]:
        all_dfs = []

        for league_dir in self.base_path.iterdir():
            if not league_dir.is_dir():
                continue

            for file_path in league_dir.glob("*_stats.csv"):
                df = self.load_csv_safe(file_path)

                if df is not None:
                    # Normalisation
                    df = normalize_column_names(df)

                    # Normalisation noms équipes
                    if "home_team" in df.columns:
                        df["home_team"] = df["home_team"].apply(normalize_team_name)

                    if "away_team" in df.columns:
                        df["away_team"] = df["away_team"].apply(normalize_team_name)

                    # Ajout saison depuis nom de fichier
                    season = file_path.stem.replace("_stats", "")
                    df["season"] = season

                    all_dfs.append(df)

                    if self.verbose:
                        print(f"✅ Chargé: {league_dir.name}/{file_path.name} ({len(df)} lignes)")

        return all_dfs

    # -----------------------------------------
    # MERGE GLOBAL
    # -----------------------------------------
    def merge_all(self):
        if self.verbose:
            print("\n Merge global (toutes ligues, toutes saisons)")

        all_dfs = self.load_all_files()

        if not all_dfs:
            print("❌ Aucun fichier trouvé")
            return None

        merged_df = pd.concat(all_dfs, ignore_index=True)

        # ============================================
        # TRI PAR DATE (smart)
        # ============================================
        if "date" in merged_df.columns:
            merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
            merged_df = merged_df.sort_values("date").reset_index(drop=True)
        else:
            print("⚠️ Colonne 'date' introuvable, tri ignoré")

        # ============================================
        # AJOUT DE LA COLONNE RESULT
        # ============================================
        merged_df = add_result_column(merged_df)

        # ============================================
        # SAUVEGARDE
        # ============================================
        output_file = self.output_dir / "dataset_raw_merged.csv"
        merged_df.to_csv(output_file, index=False)

        if self.verbose:
            print(f"\n💾 Dataset merged sauvegardé : {output_file}")
            print(f" Total lignes : {len(merged_df)}")

        return merged_df


# ============================================
# EXECUTION
# ============================================
if __name__ == "__main__":
    merger = RawMatchStatsMerger(
        base_path="data/raw/match_stats",
        verbose=True
    )
    merger.merge_all()
