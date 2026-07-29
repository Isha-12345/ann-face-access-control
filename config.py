"""
config.py
---------
Shared settings for every script in the project.
Edit paths and constants here rather than inside individual scripts.
"""

import os

# ---------- paths ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dataset_final")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(ROOT, "logs")

for _d in (MODEL_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- data ----------
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
CLASS_NAMES = ["intrusion", "known", "unknown"]  # alphabetical: Keras order
KNOWN_INDEX = CLASS_NAMES.index("known")
SEED = 42

# ---------- training ----------
EPOCHS_FROZEN = 8       # stage 1: backbone frozen
EPOCHS_FINETUNE = 6     # stage 2: top layers unfrozen
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
UNFREEZE_LAYERS = 30    # how many final backbone layers to unfreeze
DROPOUT = 0.3

# ---------- embedding model (Task 2) ----------
EMBED_DIM = 128
EMBED_EPOCHS = 12

# ---------- Raspberry Pi GPIO pins (BCM numbering) ----------
PIN_GREEN = 17
PIN_RED = 27
PIN_BUZZER = 22

# ---------- deployment behaviour ----------
CONFIDENCE_THRESHOLD = 0.60   # below this, a face is treated as unknown
STABLE_FRAMES = 3             # consecutive agreeing frames before acting
ALERT_SECONDS = 2.0           # how long an alert stays active

# ---------- CPU performance (this machine has no GPU) ----------
# TensorFlow already uses every logical core by default (intra/inter op = 0 = auto),
# so these do not unlock hidden speed - they just make the thread counts explicit
# and tunable. On a 4-core / 8-thread CPU, if hyper-thread contention hurts you can
# try physical-cores-only:  INTRA_OP_THREADS=4 python train_classifier.py ...
INTRA_OP_THREADS = int(os.environ.get("INTRA_OP_THREADS", os.cpu_count() or 4))
INTER_OP_THREADS = int(os.environ.get("INTER_OP_THREADS", 2))


def configure_cpu_threads():
    """Pin TensorFlow's CPU thread pools. Call once, before any model is built."""
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    except RuntimeError:
        pass  # TF runtime already initialised - safe to ignore
