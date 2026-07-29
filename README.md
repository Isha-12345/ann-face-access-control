# Intelligent Security and Access Control System

MSc Data Science — STW7088CEM Artificial Neural Networks
Isha Adhikari (250656)

Three-class access control (known / unknown / intrusion) using MobileNet
transfer learning and embedding-based open-set anomaly detection, deployed
on a Raspberry Pi 3B with LED and buzzer alerts.

---

## 1. Folder structure

Create this structure. Folders marked `[you create]` hold data you collect
or download; the rest are produced by the scripts.

```
ann_project/
├── config.py                  all shared settings (paths, pins, epochs)
├── tflite_utils.py            TFLite helper (float and INT8)
│
├── capture_faces.py           STEP 1  collect volunteer face images
├── build_dataset.py           STEP 2  assemble the three-class dataset
├── train_classifier.py        STEP 3  Task 1 - classification
├── train_embedding.py         STEP 4  Task 2 - open-set anomaly detection
├── evaluate.py                STEP 5  metrics, confusion matrix, ROC
├── quantize.py                STEP 6  TFLite float32 + INT8 conversion
├── benchmark_pi.py            STEP 7  latency / FPS measurement
├── deploy_pi.py               STEP 8  live system with GPIO alerts
│
├── requirements-train.txt
├── requirements-pi.txt
│
├── sources/                   [you create] downloaded Kaggle datasets
│   ├── lfw/          <person>/img.jpg
│   ├── celeba/       img_000001.jpg ...
│   ├── lfw_smfrd/    <person>/img.jpg
│   └── rmfrd/        <person>/img.jpg
│
├── dataset_raw/               [you create] enrolled volunteers
│   ├── ramesh/       ramesh_0001.jpg ...
│   └── sita/         ...
│
├── strangers_raw/             [you create] non-enrolled people, SAME camera
│   ├── stranger1/
│   └── stranger2/
│
├── hard_negatives/            [you create] enrolled volunteers WEARING MASKS
│   └── ramesh_masked/         (never used for training - test only)
│
├── dataset_final/             created by build_dataset.py
│   ├── train/{known,unknown,intrusion}/
│   ├── val/{known,unknown,intrusion}/
│   └── test/{known,unknown,intrusion}/
│
├── models/                    created automatically
│   ├── mobilenetv2.keras
│   ├── mobilenetv3.keras
│   ├── baseline.keras
│   ├── embedding.keras
│   ├── centroids.npz
│   ├── mobilenetv2_float32.tflite
│   └── mobilenetv2_int8.tflite
│
├── results/                   created automatically - REPORT FIGURES
│   ├── *_curves.png           training curves
│   ├── *_confusion.png        confusion matrices
│   ├── *_roc.png              ROC curves
│   ├── *_metrics.json
│   ├── *_quantisation.json    size / accuracy trade-off
│   ├── embedding_distances.png
│   ├── model_comparison.csv   the main comparison table
│   └── benchmark.json         Pi latency and FPS
│
└── logs/
    └── access_log.csv         timestamped decisions from the live system
```

---

## 2. Installation

### Training machine (laptop or Colab)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-train.txt
```

### Raspberry Pi

```bash
pip3 install -r requirements-pi.txt --break-system-packages
```

Only `tflite_utils.py`, `deploy_pi.py`, `benchmark_pi.py`, `config.py` and
the `.tflite` model file need to be copied to the Pi.

---

## 3. Run order

```bash
# STEP 1 - collect faces (repeat per volunteer, across 2-3 sessions)
python capture_faces.py --name ramesh --target 250
python capture_faces.py --name sita   --target 250

# strangers for the unknown class (same camera)
python capture_faces.py --name stranger1 --out strangers_raw --target 150

# STEP 2 - build the balanced three-class dataset
python build_dataset.py --per-class 10000

# STEP 3 - Task 1: train all three architectures
python train_classifier.py --model mobilenetv2
python train_classifier.py --model mobilenetv3
python train_classifier.py --model baseline

# STEP 4 - Task 2: open-set anomaly detection
python train_embedding.py

# STEP 5 - evaluate and build the comparison table
python evaluate.py --model mobilenetv2
python evaluate.py --model mobilenetv3
python evaluate.py --model baseline
python evaluate.py --compare

# STEP 6 - quantise the best model
python quantize.py --model mobilenetv2

# ---- copy models/mobilenetv2_int8.tflite to the Raspberry Pi ----

# STEP 7 - on the Pi: measure latency and FPS
python3 benchmark_pi.py --model models/mobilenetv2_int8.tflite --runs 100

# STEP 8 - on the Pi: run the live system
python3 deploy_pi.py --model models/mobilenetv2_int8.tflite
```

Test the live system on a laptop first (no hardware needed):

```bash
python deploy_pi.py --model models/mobilenetv2_int8.tflite --no-gpio
```

---

## 4. Wiring

BCM pin numbers, set in `config.py`.

| Component     | Pi pin (BCM) | Wiring                                    |
|---------------|--------------|-------------------------------------------|
| Green LED     | GPIO 17      | GPIO 17 -> 220-330 Ω resistor -> LED -> GND |
| Red LED       | GPIO 27      | GPIO 27 -> 220-330 Ω resistor -> LED -> GND |
| Active buzzer | GPIO 22      | GPIO 22 -> buzzer + , buzzer - -> GND      |

The resistors are required. The LED long leg (anode) goes toward the
resistor, the short leg (cathode) to ground.

---


