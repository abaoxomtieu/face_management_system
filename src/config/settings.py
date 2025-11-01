import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).parent.parent.parent

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 9001))

# Face Recognition Settings
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", 0.5))
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 512))

# Model Configuration
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
# Options: buffalo_l (large, most accurate), buffalo_s (small), buffalo_sc (small-compact)

# Storage Paths
FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", BASE_DIR / "face_data"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", FACE_DATA_DIR / "embeddings"))
STUDENT_IMAGES_DIR = Path(os.getenv("STUDENT_IMAGES_DIR", FACE_DATA_DIR / "student_images"))
CLASS_METADATA_FILE = Path(os.getenv("CLASS_METADATA_FILE", FACE_DATA_DIR / "classes.json"))

# Create directories if they don't exist
FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
STUDENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

