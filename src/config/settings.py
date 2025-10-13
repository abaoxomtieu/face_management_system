import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).parent.parent.parent

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 9001))
RELOAD = os.getenv("RELOAD", "False").lower() == "true"

# Face Recognition Settings
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", 0.5))
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6))
MAX_FACES_PER_IMAGE = int(os.getenv("MAX_FACES_PER_IMAGE", 50))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 512))

# Model Configuration
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
# Options: buffalo_l (large, most accurate), buffalo_s (small), buffalo_sc (small-compact)

# Storage Paths
FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", BASE_DIR / "face_data"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", FACE_DATA_DIR / "embeddings"))
STUDENT_IMAGES_DIR = Path(os.getenv("STUDENT_IMAGES_DIR", FACE_DATA_DIR / "student_images"))
METADATA_FILE = Path(os.getenv("METADATA_FILE", FACE_DATA_DIR / "metadata.json"))
CLASS_METADATA_FILE = Path(os.getenv("CLASS_METADATA_FILE", FACE_DATA_DIR / "classes.json"))

# Create directories if they don't exist
FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
STUDENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Redis Configuration (optional)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Authentication (optional)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 30))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

