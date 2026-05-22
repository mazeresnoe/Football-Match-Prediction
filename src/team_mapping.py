"""
Mapping définitif des noms d'équipes entre les 3 sources.
Sofascore est la référence commune (noms lowercase).

Structure :
    TEAM_MAPPING[sofascore_name] = {"fdc": "football-data name", "api": "Odds API name"}

    None = équipe non présente dans cette source
"""

TEAM_MAPPING = {

    # ==================== PREMIER LEAGUE ====================
    "arsenal":                 {"fdc": "Arsenal",       "api": "Arsenal"},
    "aston villa":             {"fdc": "Aston Villa",   "api": "Aston Villa"},
    "bournemouth":             {"fdc": "Bournemouth",   "api": "Bournemouth"},
    "brentford":               {"fdc": "Brentford",     "api": "Brentford"},
    "brighton & hove albion":  {"fdc": "Brighton",      "api": "Brighton and Hove Albion"},
    "burnley":                 {"fdc": "Burnley",       "api": "Burnley"},
    "cardiff city":            {"fdc": "Cardiff",       "api": None},
    "chelsea":                 {"fdc": "Chelsea",       "api": "Chelsea"},
    "crystal palace":          {"fdc": "Crystal Palace","api": "Crystal Palace"},
    "everton":                 {"fdc": "Everton",       "api": "Everton"},
    "fulham":                  {"fdc": "Fulham",        "api": "Fulham"},
    "huddersfield town":       {"fdc": "Huddersfield",  "api": None},
    "hull city":               {"fdc": "Hull",          "api": None},
    "ipswich town":            {"fdc": "Ipswich",       "api": "Ipswich Town"},
    "leeds united":            {"fdc": "Leeds",         "api": "Leeds United"},
    "leicester city":          {"fdc": "Leicester",     "api": "Leicester City"},
    "liverpool":               {"fdc": "Liverpool",     "api": "Liverpool"},
    "luton town":              {"fdc": "Luton",         "api": None},
    "manchester city":         {"fdc": "Man City",      "api": "Manchester City"},
    "manchester united":       {"fdc": "Man United",    "api": "Manchester United"},
    "middlesbrough":           {"fdc": "Middlesbrough", "api": None},
    "newcastle united":        {"fdc": "Newcastle",     "api": "Newcastle United"},
    "norwich city":            {"fdc": "Norwich",       "api": "Norwich City"},
    "nottingham forest":       {"fdc": "Nott'm Forest", "api": "Nottingham Forest"},
    "sheffield united":        {"fdc": "Sheffield United", "api": "Sheffield United"},
    "southampton":             {"fdc": "Southampton",   "api": "Southampton"},
    "stoke city":              {"fdc": "Stoke",         "api": "Stoke City"},
    "sunderland":              {"fdc": "Sunderland",    "api": "Sunderland"},
    "swansea city":            {"fdc": "Swansea",       "api": "Swansea City"},
    "tottenham hotspur":       {"fdc": "Tottenham",     "api": "Tottenham Hotspur"},
    "watford":                 {"fdc": "Watford",       "api": None},
    "west bromwich albion":    {"fdc": "West Brom",     "api": None},
    "west ham united":         {"fdc": "West Ham",      "api": "West Ham United"},
    "wolverhampton":           {"fdc": "Wolves",        "api": "Wolverhampton Wanderers"},

    # ==================== CHAMPIONSHIP ====================
    "barnsley":                {"fdc": "Barnsley",      "api": None},
    "birmingham city":         {"fdc": "Birmingham",    "api": "Birmingham City"},
    "blackburn rovers":        {"fdc": "Blackburn",     "api": "Blackburn Rovers"},
    "blackpool":               {"fdc": "Blackpool",     "api": None},
    "bolton wanderers":        {"fdc": "Bolton",        "api": None},
    "bristol city":            {"fdc": "Bristol City",  "api": "Bristol City"},
    "burton albion":           {"fdc": None,            "api": None},
    "charlton athletic":       {"fdc": "Charlton",      "api": None},
    "coventry city":           {"fdc": "Coventry",      "api": "Coventry City"},
    "derby county":            {"fdc": "Derby",         "api": "Derby County"},
    "millwall":                {"fdc": "Millwall",      "api": None},
    "oxford united":           {"fdc": "Oxford",        "api": "Oxford United"},
    "peterborough united":     {"fdc": "Peterboro",     "api": None},
    "plymouth argyle":         {"fdc": "Plymouth",      "api": None},
    "portsmouth":              {"fdc": "Portsmouth",    "api": "Portsmouth"},
    "preston north end":       {"fdc": "Preston",       "api": "Preston North End"},
    "queens park rangers":     {"fdc": "QPR",           "api": "Queens Park Rangers"},
    "reading":                 {"fdc": "Reading",       "api": None},
    "rotherham united":        {"fdc": "Rotherham",     "api": None},
    "sheffield wednesday":     {"fdc": "Sheffield Weds","api": "Sheffield Wednesday"},
    "wigan athletic":          {"fdc": "Wigan",         "api": None},
    "wrexham":                 {"fdc": None,            "api": "Wrexham AFC"},
    "wycombe wanderers":       {"fdc": "Wycombe",       "api": None},

    # ==================== LIGUE 1 ====================
    "paris saint-germain":     {"fdc": "Paris SG",      "api": "Paris Saint Germain"},
    "olympique lyonnais":      {"fdc": "Lyon",          "api": "Lyon"},
    "olympique de marseille":  {"fdc": "Marseille",     "api": "Marseille"},
    "as monaco":               {"fdc": "Monaco",        "api": "AS Monaco"},
    "lille":                   {"fdc": "Lille",         "api": "Lille"},
    "nice":                    {"fdc": "Nice",          "api": "Nice"},
    "stade rennais":           {"fdc": "Rennes",        "api": "Rennes"},
    "bordeaux":                {"fdc": "Bordeaux",      "api": None},
    "montpellier":             {"fdc": "Montpellier",   "api": None},
    "saint-étienne":           {"fdc": "St Etienne",    "api": "Saint Etienne"},
    "nantes":                  {"fdc": "Nantes",        "api": "Nantes"},
    "rc strasbourg":           {"fdc": "Strasbourg",    "api": "Strasbourg"},
    "toulouse":                {"fdc": "Toulouse",      "api": "Toulouse"},
    "stade de reims":          {"fdc": "Reims",         "api": None},
    "guingamp":                {"fdc": "Guingamp",      "api": None},
    "dijon":                   {"fdc": "Dijon",         "api": None},
    "amiens sc":               {"fdc": "Amiens",        "api": None},
    "caen":                    {"fdc": "Caen",          "api": None},
    "angers":                  {"fdc": "Angers",        "api": "Angers"},
    "nîmes olympique":         {"fdc": "Nimes",         "api": None},
    "rc lens":                 {"fdc": "Lens",          "api": "RC Lens"},
    "metz":                    {"fdc": "Metz",          "api": "Metz"},
    "stade brestois":          {"fdc": "Brest",         "api": "Brest"},
    "auxerre":                 {"fdc": "Auxerre",       "api": "Auxerre"},
    "troyes":                  {"fdc": "Troyes",        "api": "Troyes"},
    "clermont foot":           {"fdc": "Clermont",      "api": None},
    "lorient":                 {"fdc": "Lorient",       "api": "Lorient"},
    "paris fc":                {"fdc": "Paris FC",      "api": "Paris FC"},
    "le havre":                {"fdc": "Le Havre",      "api": "Le Havre"},
    "nancy":                   {"fdc": "Nancy",         "api": None},
    "bastia":                  {"fdc": "Bastia",        "api": None},
    "ajaccio":                 {"fdc": "Ajaccio",       "api": None},
    "rodez af":                {"fdc": "Rodez",         "api": None},
    "sochaux":                 {"fdc": "Sochaux",       "api": None},
    "grenoble foot 38":        {"fdc": "Grenoble",      "api": None},

    # ==================== LIGUE 2 ====================
    "usl dunkerque":           {"fdc": "Dunkerque",     "api": "USL Dunkerque"},
    "us boulogne côte-d'opale":{"fdc": "Boulogne",      "api": "Boulogne"},
    "chamois niortais":        {"fdc": "Niort",         "api": None},
    "red star fc":             {"fdc": "Red Star",      "api": None},
    "us orléans":              {"fdc": "Orleans",       "api": None},
    "châteauroux":             {"fdc": "Chateauroux",   "api": None},
    "fc chambly oise":         {"fdc": "Chambly",       "api": None},
    "pau fc":                  {"fdc": "Pau FC",        "api": None},
    "quevilly - rouen métropole": {"fdc": "Quevilly Rouen", "api": None},
    "annecy fc":               {"fdc": "Annecy",        "api": None},
    "stade lavallois":         {"fdc": "Laval",         "api": None},
    "us concarneau":           {"fdc": "Concarneau",    "api": None},
    "fc martigues":            {"fdc": "Martigues",     "api": None},
    "gfc ajaccio":             {"fdc": "Ajaccio GFCO",  "api": None},
    "le mans":                 {"fdc": "Le Mans",       "api": None},
    "as béziers":              {"fdc": "Beziers",       "api": None},
    "valenciennes":            {"fdc": "Valenciennes",  "api": None},

    # ==================== LA LIGA ====================
    "barcelona":               {"fdc": "Barcelona",     "api": "Barcelona"},
    "real madrid":             {"fdc": "Real Madrid",   "api": "Real Madrid"},
    "atlético madrid":         {"fdc": "Ath Madrid",    "api": "Atlético Madrid"},
    "sevilla":                 {"fdc": "Sevilla",       "api": "Sevilla"},
    "valencia":                {"fdc": "Valencia",      "api": "Valencia"},
    "villarreal":              {"fdc": "Villarreal",    "api": "Villarreal"},
    "real betis":              {"fdc": "Betis",         "api": "Real Betis"},
    "real sociedad":           {"fdc": "Sociedad",      "api": "Real Sociedad"},
    "athletic club":           {"fdc": "Ath Bilbao",    "api": "Athletic Bilbao"},
    "celta vigo":              {"fdc": "Celta",         "api": "Celta Vigo"},
    "espanyol":                {"fdc": "Espanol",       "api": "Espanyol"},
    "getafe":                  {"fdc": "Getafe",        "api": "Getafe"},
    "levante ud":              {"fdc": "Levante",       "api": "Levante"},
    "deportivo alavés":        {"fdc": "Alaves",        "api": "Alavés"},
    "rayo vallecano":          {"fdc": "Vallecano",     "api": "Rayo Vallecano"},
    "girona fc":               {"fdc": "Girona",        "api": "Girona"},
    "real valladolid":         {"fdc": "Valladolid",    "api": None},
    "granada":                 {"fdc": "Granada",       "api": None},
    "málaga":                  {"fdc": "Malaga",        "api": None},
    "osasuna":                 {"fdc": "Osasuna",       "api": "CA Osasuna"},
    "mallorca":                {"fdc": "Mallorca",      "api": "Mallorca"},
    "cádiz":                   {"fdc": "Cadiz",         "api": None},
    "elche":                   {"fdc": "Elche",         "api": "Elche CF"},
    "real oviedo":             {"fdc": "Oviedo",        "api": "Oviedo"},
    "las palmas":              {"fdc": "Las Palmas",    "api": None},
    "sporting gijón":          {"fdc": "Sp Gijon",      "api": "Sporting Gijón"},
    "eibar":                   {"fdc": "Eibar",         "api": None},
    "leganés":                 {"fdc": "Leganes",       "api": "Leganés"},
    "almería":                 {"fdc": "Almeria",       "api": "Almería"},
    "huesca":                  {"fdc": "Huesca",        "api": "SD Huesca"},
    "deportivo la coruña":     {"fdc": "La Coruna",     "api": "Deportivo La Coruña"},

    # ==================== LA LIGA 2 ====================
    "real zaragoza":           {"fdc": "Zaragoza",      "api": "Zaragoza"},
    "burgos club de fútbol":   {"fdc": "Burgos",        "api": "Burgos CF"},
    "cd castellón":            {"fdc": "Castellon",     "api": "CD Castellón"},
    "mirandés":                {"fdc": "Mirandes",      "api": "CD Mirandés"},
    "cultural leonesa":        {"fdc": "Cultural Leonesa","api": "Cultural Leonesa"},
    "cádiz cf":                {"fdc": "Cadiz",         "api": "Cádiz CF"},
    "córdoba":                 {"fdc": "Cordoba",       "api": "Córdoba"},
    "real racing club":        {"fdc": "Santander",     "api": "Real Racing Club de Santander"},
    "real sociedad b u21":     {"fdc": "Sociedad B",    "api": "Real Sociedad B"},
    "real valladolid cf":      {"fdc": "Valladolid",    "api": "Real Valladolid CF"},
    "fc andorra":              {"fdc": "Andorra",       "api": "Andorra CF"},
    "albacete balompié":       {"fdc": "Albacete",      "api": "Albacete"},
    "ad ceuta":                {"fdc": "Ceuta",         "api": "AD Ceuta FC"},
    "granada cf":              {"fdc": "Granada",       "api": "Granada CF"},
    "sd huesca":               {"fdc": "Huesca",        "api": "SD Huesca"},
    "málaga cf":               {"fdc": "Malaga",        "api": "Málaga"},
    "ad alcorcón":             {"fdc": "Alcorcon",      "api": None},
    "barcelona atlètic":       {"fdc": None,            "api": None},
    "cd eldense":              {"fdc": "Eldense",       "api": None},
    "cd lugo":                 {"fdc": "Lugo",          "api": None},
    "cd tenerife":             {"fdc": None,            "api": None},
    "ce sabadell":             {"fdc": None,            "api": None},
    "cf fuenlabrada":          {"fdc": "Fuenlabrada",   "api": None},
    "extremadura ud":          {"fdc": "Extremadura UD","api": None},
    "fc cartagena":            {"fdc": "Cartagena",     "api": None},
    "gimnàstic de tarragona":  {"fdc": "Gimnastic",     "api": None},
    "lorca fc":                {"fdc": None,            "api": None},
    "numancia":                {"fdc": "Numancia",      "api": None},
    "racing de ferrol":        {"fdc": "Ferrol",        "api": None},
    "rayo majadahonda":        {"fdc": None,            "api": None},
    "sd amorebieta":           {"fdc": "Amorebieta",    "api": None},
    "sd ponferradina":         {"fdc": "Ponferradina",  "api": None},
    "sevilla atlético":        {"fdc": None,            "api": None},
    "ud ibiza":                {"fdc": "Ibiza",         "api": None},
    "ud logroñés":             {"fdc": "Logrones",      "api": None},
    "villarreal cf b u23":     {"fdc": None,            "api": None},
    "ucam murcia":             {"fdc": None,            "api": None},

    # ==================== SERIE A ====================
    "juventus":                {"fdc": "Juventus",      "api": "Juventus"},
    "inter":                   {"fdc": "Inter",         "api": "Inter Milan"},
    "milan":                   {"fdc": "Milan",         "api": "AC Milan"},
    "napoli":                  {"fdc": "Napoli",        "api": "Napoli"},
    "roma":                    {"fdc": "Roma",          "api": "AS Roma"},
    "lazio":                   {"fdc": "Lazio",         "api": "Lazio"},
    "atalanta":                {"fdc": "Atalanta",      "api": "Atalanta BC"},
    "fiorentina":              {"fdc": "Fiorentina",    "api": "Fiorentina"},
    "torino":                  {"fdc": "Torino",        "api": "Torino"},
    "sampdoria":               {"fdc": "Sampdoria",     "api": "Sampdoria"},
    "genoa":                   {"fdc": "Genoa",         "api": "Genoa"},
    "bologna":                 {"fdc": "Bologna",       "api": "Bologna"},
    "sassuolo":                {"fdc": "Sassuolo",      "api": "Sassuolo"},
    "udinese":                 {"fdc": "Udinese",       "api": "Udinese"},
    "cagliari":                {"fdc": "Cagliari",      "api": "Cagliari"},
    "empoli":                  {"fdc": "Empoli",        "api": "Empoli"},
    "hellas verona":           {"fdc": "Verona",        "api": "Hellas Verona"},
    "lecce":                   {"fdc": "Lecce",         "api": "Lecce"},
    "brescia":                 {"fdc": "Brescia",       "api": None},
    "benevento":               {"fdc": "Benevento",     "api": None},
    "crotone":                 {"fdc": "Crotone",       "api": None},
    "spezia":                  {"fdc": "Spezia",        "api": None},
    "salernitana":             {"fdc": "Salernitana",   "api": None},
    "como":                    {"fdc": "Como",          "api": "Como"},
    "parma":                   {"fdc": "Parma",         "api": "Parma"},
    "venezia":                 {"fdc": "Venezia",       "api": "Venezia"},
    "cremonese":               {"fdc": "Cremonese",     "api": "Cremonese"},
    "pisa":                    {"fdc": "Pisa",          "api": "Pisa"},
    "monza":                   {"fdc": "Monza",         "api": "Monza"},
    "frosinone":               {"fdc": "Frosinone",     "api": "Frosinone"},
    "chievoverona":            {"fdc": "Chievo",        "api": None},
    "spal":                    {"fdc": "Spal",          "api": None},
    "palermo":                 {"fdc": "Palermo",       "api": "Palermo"},

    # ==================== SERIE B ====================
    "bari":                    {"fdc": "Bari",          "api": "Bari"},
    "cesena":                  {"fdc": "Cesena",        "api": "Cesena FC"},
    "juve stabia":             {"fdc": "Juve Stabia",   "api": "Juve Stabia"},
    "mantova":                 {"fdc": "Mantova",       "api": "Mantova"},
    "modena":                  {"fdc": "Modena",        "api": "Modena"},
    "padova":                  {"fdc": "Padova",        "api": "Padova"},
    "pescara":                 {"fdc": "Pescara",       "api": "Pescara"},
    "reggiana":                {"fdc": "Reggiana",      "api": "Reggiana"},
    "virtus entella":          {"fdc": "Virtus Entella","api": "Virtus Entella"},
    "us catanzaro 1929":       {"fdc": "Catanzaro",     "api": "US Catanzaro 1929"},
    "alessandria":             {"fdc": "Alessandria",   "api": None},
    "ascoli":                  {"fdc": "Ascoli",        "api": None},
    "carpi":                   {"fdc": "Carpi",         "api": None},
    "carrarese":               {"fdc": "Carrarese",     "api": None},
    "cittadella":              {"fdc": "Cittadella",    "api": None},
    "cosenza":                 {"fdc": "Cosenza",       "api": None},
    "feralpisalò":             {"fdc": "FeralpiSalo",   "api": None},
    "foggia":                  {"fdc": "Foggia",        "api": None},
    "l.r. vicenza":            {"fdc": "Vicenza",       "api": None},
    "lecco":                   {"fdc": "Lecco",         "api": None},
    "perugia":                 {"fdc": "Perugia",       "api": None},
    "pordenone":               {"fdc": "Pordenone",     "api": None},
    "reggina 1914":            {"fdc": "Reggina",       "api": None},
    "südtirol":                {"fdc": "Sudtirol",      "api": None},
    "ternana":                 {"fdc": "Ternana",       "api": None},
    "us livorno 1915":         {"fdc": "Livorno",       "api": None},
    "trapani":                 {"fdc": "Trapani",       "api": None},

    # ==================== BUNDESLIGA ====================
    "fc bayern münchen":       {"fdc": "Bayern Munich", "api": "Bayern Munich"},
    "borussia dortmund":       {"fdc": "Dortmund",      "api": "Borussia Dortmund"},
    "rb leipzig":              {"fdc": "RB Leipzig",    "api": "RB Leipzig"},
    "borussia m'gladbach":     {"fdc": "M'gladbach",    "api": "Borussia Monchengladbach"},
    "bayer 04 leverkusen":     {"fdc": "Leverkusen",    "api": "Bayer Leverkusen"},
    "fc schalke 04":           {"fdc": "Schalke 04",    "api": "FC Schalke 04"},
    "tsg hoffenheim":          {"fdc": "Hoffenheim",    "api": "TSG Hoffenheim"},
    "vfl wolfsburg":           {"fdc": "Wolfsburg",     "api": "VfL Wolfsburg"},
    "eintracht frankfurt":     {"fdc": "Ein Frankfurt", "api": "Eintracht Frankfurt"},
    "sv werder bremen":        {"fdc": "Werder Bremen", "api": "Werder Bremen"},
    "hertha bsc":              {"fdc": "Hertha",        "api": None},
    "1. fsv mainz 05":         {"fdc": "Mainz",         "api": "FSV Mainz 05"},
    "sc freiburg":             {"fdc": "Freiburg",      "api": "SC Freiburg"},
    "fc augsburg":             {"fdc": "Augsburg",      "api": "Augsburg"},
    "vfb stuttgart":           {"fdc": "Stuttgart",     "api": "VfB Stuttgart"},
    "hannover 96":             {"fdc": "Hannover",      "api": None},
    "hamburger sv":            {"fdc": "Hamburg",       "api": "Hamburger SV"},
    "1. fc köln":              {"fdc": "FC Koln",       "api": "1. FC Köln"},
    "fc st. pauli":            {"fdc": "St Pauli",      "api": "FC St. Pauli"},
    "1. fc heidenheim":        {"fdc": "Heidenheim",    "api": "1. FC Heidenheim"},
    "1. fc union berlin":      {"fdc": "Union Berlin",  "api": "Union Berlin"},
    "fc ingolstadt 04":        {"fdc": "Ingolstadt",    "api": None},
    "fortuna düsseldorf":      {"fdc": "Fortuna Dusseldorf", "api": None},
    "arminia bielefeld":       {"fdc": "Bielefeld",     "api": "Arminia Bielefeld"},
    "darmstadt 98":            {"fdc": "Darmstadt",     "api": "SV Darmstadt 98"},
    "holstein kiel":           {"fdc": "Holstein Kiel", "api": "Holstein Kiel"},
    "sc paderborn 07":         {"fdc": "Paderborn",     "api": "SC Paderborn"},

    # ==================== 2. BUNDESLIGA ====================
    "vfl bochum 1848":         {"fdc": "Bochum",        "api": "VfL Bochum"},
    "spvgg greuther fürth":    {"fdc": "Greuther Furth","api": "Greuther Fürth"},
    "sg dynamo dresden":       {"fdc": "Dresden",       "api": "Dynamo Dresden"},
    "1. fc magdeburg":         {"fdc": "Magdeburg",     "api": "1. FC Magdeburg"},
    "1. fc kaiserslautern":    {"fdc": "Kaiserslautern","api": "1. FC Kaiserslautern"},
    "f.c. hansa rostock":      {"fdc": "Hansa Rostock", "api": None},
    "sv 07 elversberg":        {"fdc": "Elversberg",    "api": "Elversberg"},
    "eintracht braunschweig":  {"fdc": "Braunschweig",  "api": "Eintracht Braunschweig"},
    "1. fc nürnberg":          {"fdc": "Nurnberg",      "api": "1. FC Nürnberg"},
    "1. fc saarbrücken":       {"fdc": None,            "api": None},
    "erzgebirge aue":          {"fdc": "Erzgebirge Aue","api": None},
    "karlsruher sc":           {"fdc": "Karlsruhe",     "api": None},
    "msv duisburg":            {"fdc": "Duisburg",      "api": None},
    "preußen münster":         {"fdc": "PreuÃen MÃ¼nster", "api": None},
    "ssv jahn regensburg":     {"fdc": "Regensburg",    "api": None},
    "ssv ulm 1846":            {"fdc": None,            "api": None},
    "sv sandhausen":           {"fdc": "Sandhausen",    "api": None},
    "sv wehen wiesbaden":      {"fdc": "Wehen",         "api": None},
    "tsv 1860 münchen":        {"fdc": None,            "api": None},
    "vfl osnabrück":           {"fdc": "Osnabruck",     "api": None},
    "fc würzburger kickers":   {"fdc": "Wurzburger Kickers", "api": None},

    # ==================== COUPES EUROPEENNES ====================
    "sporting braga":          {"fdc": None,            "api": "SC Braga"},
    "shakhtar donetsk":        {"fdc": None,            "api": "Shakhtar Donetsk"},
}


# ============================================================
# REVERSE MAPPINGS — normalise chaque source vers Sofascore
# ============================================================

def build_reverse(source_key: str) -> dict:
    reverse = {}
    for sofascore_name, aliases in TEAM_MAPPING.items():
        alias = aliases.get(source_key)
        if alias:
            reverse[alias] = sofascore_name
    return reverse

FDC_TO_SOFASCORE = build_reverse("fdc")
API_TO_SOFASCORE = build_reverse("api")


def normalize_team_names(df, source: str, home_col="home_team", away_col="away_team"):
    """
    Normalise les noms d'équipes d'un DataFrame vers les noms Sofascore.
    source : "fdc" ou "api"
    """
    mapping = FDC_TO_SOFASCORE if source == "fdc" else API_TO_SOFASCORE
    df = df.copy()
    df[home_col] = df[home_col].map(mapping).fillna(df[home_col])
    df[away_col] = df[away_col].map(mapping).fillna(df[away_col])
    return df


# ============================================================
# DIAGNOSTIC — lance directement pour voir les trous
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    print("=" * 60)
    print("DIAGNOSTIC MAPPING")
    print("=" * 60)

    # --- Odds API (futurs) ---
    api_path = Path("data/futur_odds/upcoming_odds.csv")
    if api_path.exists():
        df_api = pd.read_csv(api_path)
        all_api = set(df_api["home_team"]) | set(df_api["away_team"])
        unmapped = sorted([t for t in all_api if t not in API_TO_SOFASCORE])
        print(f"\n[Odds API] {len(all_api)} équipes — Non mappées ({len(unmapped)}) :")
        for t in unmapped: print(f"  - {t}")
        df_norm = normalize_team_names(df_api, "api")
        df_norm.to_csv("data/futur_odds/upcoming_odds_normalized.csv", index=False)
        print(f"  ✓ Sauvegardé : data/futur_odds/upcoming_odds_normalized.csv")

    # --- football-data (historiques) ---
    fdc_path = Path("data/odds/all_odds_clean.csv")
    if fdc_path.exists():
        df_fdc = pd.read_csv(fdc_path, encoding="latin-1")
        all_fdc = set(df_fdc["home_team"]) | set(df_fdc["away_team"])
        unmapped = sorted([t for t in all_fdc if t not in FDC_TO_SOFASCORE])
        print(f"\n[football-data] {len(all_fdc)} équipes — Non mappées ({len(unmapped)}) :")
        for t in unmapped: print(f"  - {t}")
        df_norm = normalize_team_names(df_fdc, "fdc")
        df_norm.to_csv("data/odds/all_odds_standardized.csv", index=False)
        print(f"  ✓ Sauvegardé : data/odds/all_odds_standardized.csv")

    print("\n" + "=" * 60)
    print("Remplace les anciens fichiers par les versions _normalized")
    print("=" * 60)