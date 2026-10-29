import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# Database configuration
DATABASE_PATH = BASE_DIR / "facetrack.db"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Computer Vision & Models
MODELS_DIR = BASE_DIR / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Face Recognition Settings
# Cosine similarity threshold for SFace: default >= 0.363 is official threshold, 0.40 recommended for high confidence
COSINE_SIMILARITY_THRESHOLD = 0.40
FACE_SCORE_THRESHOLD = 0.70  # YuNet detection score
NMS_THRESHOLD = 0.30

# Venue & Tracking Configuration
DEFAULT_VENUE_NAME = "Apex Elite Fitness & Club"
DEFAULT_ZONES = [
    {"id": "main_entrance", "name": "Main Entrance & Reception", "description": "Entry turnstile gate"},
    {"id": "cardio_zone", "name": "Cardio & Treadmills Zone", "description": "Treadmills and cycle area"},
    {"id": "weight_room", "name": "Free Weights & Strength Area", "description": "Dumbbells and power racks"},
    {"id": "vip_lounge", "name": "VIP Members Lounge", "description": "Private lounge and recovery room"},
    {"id": "studio_area", "name": "Group Studio & Yoga Floor", "description": "Aerobics and yoga studio"}
]

# Dwell Time Parameters (in seconds)
MIN_DWELL_SECONDS = 5       # Minimum duration to count as an intentional visit
DWELL_TIMEOUT_SECONDS = 15   # Max inactivity before closing a visit session

# DPDP Act Compliance Settings
RETENTION_PERIOD_DAYS = 90  # Automated log pruning window
CONSENT_VERSION = "v1.2-2026-DPDP"
DATA_FIDUCIARY_NAME = "FaceTrack AI Partner Venue"
