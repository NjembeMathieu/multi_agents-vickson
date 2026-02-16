"""
Configuration centrale du système multi-agents
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins du projet
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "Corpus"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
OUTPUT_DIR = BASE_DIR / "output"

# Créer les dossiers s'ils n'existent pas
VECTORSTORE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration du modèle Gemini
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 2035

# Configuration de l'embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Seuils de validation par cycle
VALIDATION_THRESHOLDS = {
    "Primaire": 90,
    "Secondaire": 85,
    "Universitaire": 80
}

# Seuil de similarité pour réutilisation
SIMILARITY_THRESHOLD = 0.90

# Limite de boucles de correction
MAX_CORRECTION_LOOPS = 3

# Configuration des gabarits par durée
DURATION_TEMPLATES = {
    "1-2h": "court",
    "3-4h": "moyen",
    "5h+": "etendu"
}

# Matières supportées dans le Corpus
SUPPORTED_SUBJECTS = ["Informatique", "Mathématiques"]

# Niveaux d'enseignement
EDUCATION_LEVELS = {
    "Primaire": ["CP", "CE1", "CE2", "CM1", "CM2"],
    "Secondaire": ["6ème", "5ème", "4ème", "3ème", "2nde", "1ère", "Terminale"],
    "Universitaire": ["Licence 1", "Licence 2", "Licence 3", "Master 1", "Master 2"]
}