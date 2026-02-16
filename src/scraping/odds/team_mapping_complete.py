"""
Script pour créer un nouveau fichier odds avec les noms d'équipes standardisés

Usage:
    python create_odds_mapping.py
"""

import pandas as pd
from pathlib import Path

# ========================================
# MAPPING COMPLET DES NOMS D'ÉQUIPES
# ========================================

TEAM_NAME_MAPPING = {
    # ===== PREMIER LEAGUE =====
    'Arsenal': 'arsenal',
    'Leicester': 'leicester city',
    'Brighton': 'brighton & hove albion',
    'Man City': 'manchester city',
    'Chelsea': 'chelsea',
    'Burnley': 'burnley',
    'Crystal Palace': 'crystal palace',
    'Huddersfield': 'huddersfield town',
    'Everton': 'everton',
    'Stoke': 'stoke city',
    'Southampton': 'southampton',
    'Swansea': 'swansea city',
    'Watford': 'watford',
    'Liverpool': 'liverpool',
    'West Brom': 'west bromwich albion',
    'Bournemouth': 'bournemouth',
    'Man United': 'manchester united',
    'West Ham': 'west ham united',
    'Newcastle': 'newcastle united',
    'Tottenham': 'tottenham hotspur',
    'Wolves': 'wolverhampton',
    'Cardiff': 'cardiff city',
    'Fulham': 'fulham',
    'Aston Villa': 'aston villa',
    
    # ===== CHAMPIONSHIP =====
    'Reading': 'reading',
    'Derby': 'derby county',
    'Wigan': 'wigan athletic',
    'Sheffield Weds': 'sheffield wednesday',
    'Bolton': 'bolton wanderers',
    'Sheffield United': 'sheffield united',
    'Preston': 'preston north end',
    'QPR': 'queens park rangers',
    'Millwall': 'millwall',
    'Middlesbrough': 'middlesbrough',
    'Ipswich': 'ipswich town',
    'Blackburn': 'blackburn rovers',
    'Bristol City': 'bristol city',
    "Nott'm Forest": 'nottingham forest',
    'Brentford': 'brentford',
    'Rotherham': 'rotherham united',
    'Birmingham': 'birmingham city',
    'Norwich': 'norwich city',
    'Leeds': 'leeds united',
    'Hull': 'hull city',
    'Sunderland': 'sunderland',
    'Barnsley': 'barnsley',
    'Charlton': 'charlton athletic',
    'Luton': 'luton town',
    'Coventry': 'coventry city',
    'Wycombe': 'wycombe wanderers',
    'Peterboro': 'peterborough united',
    'Blackpool': 'blackpool',
    'Plymouth': 'plymouth argyle',
    'Portsmouth': 'portsmouth',
    'Oxford': 'oxford united',
    
    # ===== LA LIGA =====
    'Barcelona': 'barcelona',
    'Real Madrid': 'real madrid',
    'Ath Madrid': 'atlético madrid',
    'Sevilla': 'sevilla',
    'Valencia': 'valencia',
    'Villarreal': 'villarreal',
    'Betis': 'real betis',
    'Sociedad': 'real sociedad',
    'Ath Bilbao': 'athletic club',
    'Celta': 'celta vigo',
    'Espanol': 'espanyol',
    'Getafe': 'getafe',
    'Levante': 'levante ud',
    'Alaves': 'deportivo alavés',
    'Eibar': 'eibar',
    'Leganes': 'leganés',
    'Vallecano': 'rayo vallecano',
    'Girona': 'girona fc',
    'Valladolid': 'real valladolid',
    'Huesca': 'huesca',
    'Las Palmas': 'las palmas',
    'La Coruna': 'deportivo la coruña',
    'Sp Gijon': 'sporting gijón',
    'Granada': 'granada',
    'Malaga': 'málaga',
    'Osasuna': 'osasuna',
    'Mallorca': 'mallorca',
    'Cadiz': 'cádiz',
    'Elche': 'elche',
    
    # ===== LA LIGA 2 =====
    'Almeria': 'almería',
    'Albacete': 'albacete balompié',
    'Alcorcon': 'ad alcorcón',
    'Cordoba': 'córdoba',
    'Numancia': 'numancia',
    'Lugo': 'cd lugo',
    'Zaragoza': 'real zaragoza',
    'Rayo Majadahonda': 'rayo majadahonda',
    'Oviedo': 'real oviedo',
    'Extremadura UD': 'extremadura ud',
    'Reus Deportiu': 'reus deportiu',
    'Gimnastic': 'gimnàstic de tarragona',
    'Tenerife': 'cd tenerife',
    'Mirandes': 'mirandés',
    'Fuenlabrada': 'cf fuenlabrada',
    'Santander': 'real racing club',
    'Ponferradina': 'sd ponferradina',
    'Logrones': 'ud logroñés',
    'Castellon': 'cd castellón',
    'Cartagena': 'fc cartagena',
    'Sabadell': 'ce sabadell',
    'Ibiza': 'ud ibiza',
    'Sociedad B': 'real sociedad b u21',
    'Amorebieta': 'sd amorebieta',
    'Burgos': 'burgos club de fútbol',
    'Villarreal B': 'villarreal cf b u23',
    'Andorra': 'fc andorra',
    'Eldense': 'cd eldense',
    'Ferrol': 'racing de ferrol',
    
    # ===== SERIE A =====
    'Juventus': 'juventus',
    'Inter': 'inter',
    'Milan': 'milan',
    'Napoli': 'napoli',
    'Roma': 'roma',
    'Lazio': 'lazio',
    'Atalanta': 'atalanta',
    'Fiorentina': 'fiorentina',
    'Torino': 'torino',
    'Sampdoria': 'sampdoria',
    'Genoa': 'genoa',
    'Bologna': 'bologna',
    'Sassuolo': 'sassuolo',
    'Udinese': 'udinese',
    'Cagliari': 'cagliari',
    'Empoli': 'empoli',
    'Spal': 'spal',
    'Chievo': 'chievoverona',
    'Crotone': 'crotone',
    'Verona': 'hellas verona',
    'Benevento': 'benevento',
    'Lecce': 'lecce',
    'Brescia': 'brescia',
    
    # ===== SERIE B =====
    'Parma': 'parma',
    'Frosinone': 'frosinone',
    'Palermo': 'palermo',
    'Venezia': 'venezia',
    'Spezia': 'spezia',
    'Pescara': 'pescara',
    'Cremonese': 'cremonese',
    'Salernitana': 'salernitana',
    'Foggia': 'foggia',
    'Carpi': 'carpi',
    'Cittadella': 'cittadella',
    'Ascoli': 'ascoli',
    'Cosenza': 'cosenza',
    'Padova': 'padova',
    'Livorno': 'us livorno 1915',
    'Perugia': 'perugia',
    'Pisa': 'pisa',
    'Virtus Entella': 'virtus entella',
    'Trapani': 'trapani',
    'Juve Stabia': 'juve stabia',
    'Pordenone': 'pordenone',
    'Reggina': 'reggina 1914',
    'Reggiana': 'reggiana',
    'Monza': 'monza',
    'Vicenza': 'l.r. vicenza',
    'Ternana': 'ternana',
    'Alessandria': 'alessandria',
    'Como': 'como',
    'Bari': 'bari',
    'Modena': 'modena',
    'Sudtirol': 'südtirol',
    'Catanzaro': 'catanzaro',
    'FeralpiSalo': 'feralpisalò',
    'Lecco': 'lecco',
    'Cesena': 'cesena',
    'Carrarese': 'carrarese',
    'Mantova': 'mantova',
    
    # ===== LIGUE 1 =====
    'Paris SG': 'paris saint-germain',
    'Lyon': 'olympique lyonnais',
    'Marseille': 'olympique de marseille',
    'Monaco': 'as monaco',
    'Lille': 'lille',
    'Nice': 'nice',
    'Rennes': 'stade rennais',
    'Bordeaux': 'bordeaux',
    'Montpellier': 'montpellier',
    'St Etienne': 'saint-étienne',
    'Nantes': 'nantes',
    'Strasbourg': 'rc strasbourg',
    'Toulouse': 'toulouse',
    'Reims': 'stade de reims',
    'Guingamp': 'guingamp',
    'Dijon': 'dijon',
    'Amiens': 'amiens sc',
    'Caen': 'caen',
    'Angers': 'angers',
    'Nimes': 'nîmes olympique',
    
    # ===== LIGUE 2 =====
    'Lorient': 'lorient',
    'Lens': 'rc lens',
    'Metz': 'metz',
    'Ajaccio': 'ajaccio',
    'Ajaccio GFCO': 'gfc ajaccio',
    'Auxerre': 'auxerre',
    'Troyes': 'troyes',
    'Clermont': 'clermont foot',
    'Brest': 'stade brestois',
    'Sochaux': 'sochaux',
    'Nancy': 'nancy',
    'Valenciennes': 'valenciennes',
    'Paris FC': 'paris fc',
    'Le Havre': 'le havre',
    'Niort': 'chamois niortais',
    'Red Star': 'red star fc',
    'Orleans': 'us orléans',
    'Chateauroux': 'châteauroux',
    'Grenoble': 'grenoble foot 38',
    'Beziers': 'as béziers',
    'Chambly': 'fc chambly oise',
    'Rodez': 'rodez af',
    'Le Mans': 'le mans',
    'Dunkerque': 'usl dunkerque',
    'Pau FC': 'pau fc',
    'Bastia': 'bastia',
    'Quevilly Rouen': 'quevilly - rouen métropole',
    'Annecy': 'annecy fc',
    'Laval': 'stade lavallois',
    'Concarneau': 'us concarneau',
    'Martigues': 'fc martigues',
    
    # ===== BUNDESLIGA =====
    'Bayern Munich': 'fc bayern münchen',
    'Dortmund': 'borussia dortmund',
    'RB Leipzig': 'rb leipzig',
    "M'gladbach": "borussia m'gladbach",
    'Leverkusen': 'bayer 04 leverkusen',
    'Schalke 04': 'fc schalke 04',
    'Hoffenheim': 'tsg hoffenheim',
    'Wolfsburg': 'vfl wolfsburg',
    'Ein Frankfurt': 'eintracht frankfurt',
    'Werder Bremen': 'sv werder bremen',
    'Hertha': 'hertha bsc',
    'Mainz': '1. fsv mainz 05',
    'Freiburg': 'sc freiburg',
    'Augsburg': 'fc augsburg',
    'Stuttgart': 'vfb stuttgart',
    'Hannover': 'hannover 96',
    'Hamburg': 'hamburger sv',
    'FC Koln': '1. fc köln',
    'Nurnberg': '1. fc nürnberg',
    
    # ===== 2. BUNDESLIGA =====
    'Union Berlin': '1. fc union berlin',
    'Bochum': 'vfl bochum 1848',
    'Holstein Kiel': 'holstein kiel',
    'Heidenheim': '1. fc heidenheim',
    'Paderborn': 'sc paderborn 07',
    'St Pauli': 'fc st. pauli',
    'Darmstadt': 'darmstadt 98',
    'Ingolstadt': 'fc ingolstadt 04',
    'Regensburg': 'ssv jahn regensburg',
    'Sandhausen': 'sv sandhausen',
    'Greuther Furth': 'spvgg greuther fürth',
    'Erzgebirge Aue': 'erzgebirge aue',
    'Dresden': 'sg dynamo dresden',
    'Duisburg': 'msv duisburg',
    'Magdeburg': '1. fc magdeburg',
    'Bielefeld': 'arminia bielefeld',
    'Fortuna Dusseldorf': 'fortuna düsseldorf',
    'Wurzburger Kickers': 'fc würzburger kickers',
    'Braunschweig': 'eintracht braunschweig',
    'Karlsruhe': 'karlsruher sc',
    'Osnabruck': 'vfl osnabrück',
    'Wehen': 'sv wehen wiesbaden',
    'Kaiserslautern': '1. fc kaiserslautern',
    'Hansa Rostock': 'f.c. hansa rostock',
    'Elversberg': 'sv 07 elversberg',
    'Ulm': 'ssv ulm 1846',
    'PreuÃ\x9fen MÃ¼nster': 'preußen münster',  # Caractères spéciaux
}


def create_standardized_odds_file():
    """
    Charge le fichier odds, applique le mapping, et crée un nouveau fichier
    """
    print("="*70)
    print("  CRÉATION DU FICHIER ODDS STANDARDISÉ")
    print("="*70)
    
    # Charger le fichier odds original
    odds_path = Path("data/odds/all_odds_clean.csv")
    
    if not odds_path.exists():
        print(f"\n❌ Fichier introuvable : {odds_path}")
        return
    
    print(f"\n📂 Chargement de {odds_path}...")
    odds = pd.read_csv(odds_path)
    print(f"✓ {len(odds):,} matchs chargés")
    
    # Afficher les colonnes
    print(f"\n📋 Colonnes disponibles : {list(odds.columns)}")
    
    # Appliquer le mapping
    print(f"\n🔄 Application du mapping...")
    
    # Compter combien de matchs seront affectés
    n_home_mapped = odds['home_team'].isin(TEAM_NAME_MAPPING.keys()).sum()
    n_away_mapped = odds['away_team'].isin(TEAM_NAME_MAPPING.keys()).sum()
    
    print(f"   • Home teams à mapper : {n_home_mapped:,} / {len(odds):,} ({n_home_mapped/len(odds)*100:.1f}%)")
    print(f"   • Away teams à mapper : {n_away_mapped:,} / {len(odds):,} ({n_away_mapped/len(odds)*100:.1f}%)")
    
    # Appliquer le mapping
    odds['home_team'] = odds['home_team'].map(TEAM_NAME_MAPPING).fillna(odds['home_team'])
    odds['away_team'] = odds['away_team'].map(TEAM_NAME_MAPPING).fillna(odds['away_team'])
    
    # Équipes non mappées
    home_unmapped = odds[~odds['home_team'].isin(TEAM_NAME_MAPPING.values())]['home_team'].unique()
    away_unmapped = odds[~odds['away_team'].isin(TEAM_NAME_MAPPING.values())]['away_team'].unique()
    
    all_unmapped = set(list(home_unmapped) + list(away_unmapped))
    
    if len(all_unmapped) > 0:
        print(f"\n⚠️  {len(all_unmapped)} équipes NON mappées (probablement déjà en bon format) :")
        for team in sorted(all_unmapped)[:20]:  # Afficher les 20 premières
            print(f"   • {team}")
        if len(all_unmapped) > 20:
            print(f"   ... et {len(all_unmapped) - 20} autres")
    
    # Sauvegarder
    output_path = Path("data/odds/all_odds_standardized.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    odds.to_csv(output_path, index=False)
    
    print(f"\n✅ Fichier standardisé créé : {output_path}")
    print(f"   • {len(odds):,} matchs")
    print(f"   • Colonnes : {list(odds.columns)}")
    
    # Statistiques finales
    print(f"\n📊 Statistiques :")
    print(f"   • Équipes uniques (home) : {odds['home_team'].nunique()}")
    print(f"   • Équipes uniques (away) : {odds['away_team'].nunique()}")
    print(f"   • Équipes uniques (total) : {len(set(odds['home_team'].unique()) | set(odds['away_team'].unique()))}")
    
    return odds


def test_merge():
    """
    Test le merge entre le dataset no_xg et le nouveau fichier odds
    """
    print("\n" + "="*70)
    print("  TEST DU MERGE")
    print("="*70)
    
    # Charger les deux fichiers
    df_path = Path("data/clean/prematch/etape3/full_dataset_no_xg_clean_v2.csv")
    odds_path = Path("data/odds/all_odds_standardized.csv")
    
    if not df_path.exists() or not odds_path.exists():
        print("❌ Fichiers manquants pour le test")
        return
    
    print(f"\n📂 Chargement des fichiers...")
    df = pd.read_csv(df_path)
    odds = pd.read_csv(odds_path)
    
    print(f"✓ Dataset principal : {len(df):,} matchs")
    print(f"✓ Odds : {len(odds):,} matchs")
    
    # Convertir dates
    df['date'] = pd.to_datetime(df['date'])
    odds['date'] = pd.to_datetime(odds['date'])
    
    # Merge
    print(f"\n🔄 Merge en cours...")
    merged = df.merge(
        odds[['date', 'home_team', 'away_team', 'odds_home', 'odds_draw', 'odds_away']],
        on=['date', 'home_team', 'away_team'],
        how='left',
        indicator=True
    )
    
    # Statistiques
    n_both = (merged['_merge'] == 'both').sum()
    n_left_only = (merged['_merge'] == 'left_only').sum()
    
    print(f"\n📊 Résultats du merge :")
    print(f"   • Matchs avec odds trouvées : {n_both:,} ({n_both/len(df)*100:.1f}%)")
    print(f"   • Matchs sans odds : {n_left_only:,} ({n_left_only/len(df)*100:.1f}%)")
    
    if n_both > 0:
        print(f"\n✅ SUCCÈS ! Le merge fonctionne.")
        print(f"   Tu peux maintenant utiliser 'all_odds_standardized.csv'")
    else:
        print(f"\n⚠️ PROBLÈME : Aucun match n'a été mergé.")
        print(f"   Vérifie les noms d'équipes et les dates.")


if __name__ == "__main__":
    # 1. Créer le fichier standardisé
    odds_df = create_standardized_odds_file()
    
    # 2. Tester le merge
    test_merge()
    
    print("\n" + "="*70)
    print("  TERMINÉ")
    print("="*70)
    print(f"\n💡 Prochaine étape :")
    print(f"   1. Vérifie le fichier : data/odds/all_odds_standardized.csv")
    print(f"   2. Modifie configs.py :")
    print(f"      DATA_ODDS = DATA_DIR / 'odds/all_odds_standardized.csv'")
    print(f"   3. Relance test_baselines.py")
