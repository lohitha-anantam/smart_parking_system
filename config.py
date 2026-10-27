from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_SLOTS_FILE = PROJECT_DIR / "slots.json"
SAMPLE_SLOTS_FILE = PROJECT_DIR / "slots.example.json"
OUTPUT_DIR = PROJECT_DIR / "output"

# COCO IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# A slot is marked occupied when a vehicle box intersects at least this
# fraction of the slot polygon, or the vehicle's bottom-center is inside it.
SLOT_OVERLAP_THRESHOLD = 0.12
