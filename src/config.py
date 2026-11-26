from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_CSV = PROCESSED_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "vibration_fault_model.joblib"
TARGET_COLUMN = "type"

COLUMNS = [
    "point",
    "time",
    "trigger",
    "acc_de_h",
    "acc_de_v",
    "acc_nde_h",
    "acc_nde_v",
    "vel_de_h",
    "vel_de_v",
    "vel_nde_h",
    "vel_nde_v",
]