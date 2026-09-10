# config.py
import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/door-watchman/.env"))

MODELS_DIR = os.path.expanduser("~/door-watchman/models/engines")
SCRFD_ENGINE = os.path.join(MODELS_DIR, "scrfd_10g_fp16.engine")
ARCFACE_ENGINE = os.path.join(MODELS_DIR, "arcface_r50_fp16.engine")

DB_PATH = os.path.expanduser("~/door-watchman/data/faces.json")
PHOTOS_DIR = os.path.expanduser("~/door-watchman/enroll_photos")

GO2RTC_STREAM = "rtsp://localhost:8554/front_door"
CHECK_INTERVAL = 1.0  # seconds between recognition checks

CLIP_ENGINE = os.path.join(MODELS_DIR, "clip_image_encoder_fp16.engine")
CLIP_TEXT_EMBEDDINGS_PATH = os.path.expanduser("~/door-watchman/data/clip_text_embeddings.npz")

PIPER_DIR = os.path.expanduser("~/door-watchman/models/piper")
PIPER_MODEL = os.path.join(PIPER_DIR, "en_US-lessac-medium.onnx")

NTFY_TOPIC = "door-watchman-anish-7f3k2m"

HA_URL = "http://192.168.7.69:8123"
#ALEXA_ENTITY_ID = "media_player.anish_echo_show"
HA_TOKEN = os.environ["HA_TOKEN"]

VISITOR_PHOTOS_DIR = os.path.expanduser("~/door-watchman/data/visitor_photos")
#ALEXA_ENTITIES = ["media_player.anish_echo_show_speak", "media_player.anish_s_echo_speak"]
#ALEXA_NOTIFY_ENTITIES = ["notify.anish_echo_show_speak", "notify.anish_s_echo_speak"]
ALEXA_NOTIFY_ENTITIES = ["notify.anish_echo_show_speak"]