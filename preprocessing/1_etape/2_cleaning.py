# ============================================
# CLEANING STRUCTUREL MINIMAL - POST MATCH
# ============================================

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================
# CONFIG
# ============================================

RAW_FILE = "data/raw/merged/dataset_raw_merged.csv"
OUTPUT_FILE = "data/clean/post_match/post_match_clean.csv"

REPORT_DIR = "rapport/cleaning_rapport"
REPORT_FILE = f"{REPORT_DIR}/post_match_cleaning_report.txt"

# ============================================
# UTILS
# ============================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    if "home_team" in df.columns:
        df["home_team"] = (
            df["home_team"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "away_team" in df.columns:
        df["away_team"] = (
            df["away_team"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    priority_cols = [
        "event_id",
        "date",
        "league",
        "season",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ]

    cols = df.columns.tolist()
    ordered = [c for c in priority_cols if c in cols]
    remaining = [c for c in cols if c not in ordered]

    return df[ordered + remaining]

# ============================================
# CLEANING PIPELINE
# ============================================

def main():
    print(" Chargement dataset brut...")
    df = pd.read_csv(RAW_FILE, low_memory=False)
    print(f"✅ Shape initiale: {df.shape}")

    # ---- Normalisation minimale des colonnes
    df = normalize_columns(df)

    # ---- Drop colonnes 100% vides
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f" Colonnes supprimées (100% vides) : {len(empty_cols)}")

    # ---- Suppression des matchs annulés / invalides
    if "status" in df.columns:
        before = len(df)
        df = df[~df["status"].astype(str).str.lower().isin(["canceled", "cancelled", "postponed"])]
        print(f" Matchs annulés supprimés : {before - len(df)}")

    # ---- Conversion types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ---- Conversion automatique des colonnes numériques
    for col in df.columns:
        if col in ["home_team", "away_team", "league", "season", "description", "source_file"]:
            continue

        df[col] = pd.to_numeric(df[col], errors="ignore")

    # ---- Normalisation minimale des équipes
    df = normalize_team_names(df)

    # ---- Suppression des doublons stricts
    before = len(df)
    df = df.drop_duplicates()
    print(f" Doublons supprimés : {before - len(df)}")

    # ---- Réorganisation des colonnes
    df = reorder_columns(df)

    # ============================================
    # SAVE
    # ============================================

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ CLEANING STRUCTUREL TERMINÉ")
    print(f" Fichier sauvegardé : {OUTPUT_FILE}")
    print(f" Shape final : {df.shape}")


if __name__ == "__main__":
    main()
