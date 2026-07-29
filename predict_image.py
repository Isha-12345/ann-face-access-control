"""
predict_image.py
----------------
Test a trained classifier on ANY still image (or a folder of images) that the
model has never seen - e.g. a fresh selfie taken today. This is the honest,
real-world check that the held-out test split cannot give you, because the test
split shares sessions/lighting with training.

Usage:
    python predict_image.py path/to/photo.jpg
    python predict_image.py path/to/folder/          # every image in a folder
    python predict_image.py photo.jpg --model models/mobilenetv2_int8.tflite

The face is detected and cropped exactly like build_dataset.py, so the input
matches what the model was trained on.
"""
import argparse
import glob
import os

import cv2
import numpy as np

import config as C

MARGIN = 0.25


def to_square(img, size):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    img = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def crop_face(img, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces):
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        mx, my = int(w * MARGIN), int(h * MARGIN)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
        return to_square(img[y0:y1, x0:x1], size), True
    return to_square(img, size), False          # fallback: whole image


def load_keras(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)

    def predict(rgb):                            # rgb 0-255, HxWx3
        p = model.predict(np.expand_dims(rgb, 0), verbose=0)[0]
        return np.asarray(p, np.float32)
    return predict, C.IMG_SIZE


def load_tflite(path):
    from tflite_utils import TFLiteModel
    m = TFLiteModel(path)
    return (lambda rgb: m.predict(rgb)), m.input_size[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image file or a folder of images")
    ap.add_argument("--model", default=os.path.join(C.MODEL_DIR, "mobilenetv2.keras"))
    ap.add_argument("--threshold", type=float, default=C.CONFIDENCE_THRESHOLD,
                    help="known below this confidence is downgraded to unknown")
    args = ap.parse_args()

    if args.model.endswith(".tflite"):
        predict, size = load_tflite(args.model)
    else:
        predict, size = load_keras(args.model)

    if os.path.isdir(args.image):
        paths = sorted(sum([glob.glob(os.path.join(args.image, e))
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG")], []))
    else:
        paths = [args.image]
    if not paths:
        raise SystemExit(f"[!] no images found at {args.image}")

    print(f"[i] model: {os.path.basename(args.model)} | {len(paths)} image(s)\n")
    print(f"{'image':40s} {'prediction':12s} {'conf':>6s}  probs(intr/known/unk)")
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{os.path.basename(p):40s} <could not read>")
            continue
        face, found = crop_face(img, size)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        probs = predict(rgb)
        idx = int(np.argmax(probs))
        label = C.CLASS_NAMES[idx]
        conf = float(probs[idx])
        if label == "known" and conf < args.threshold:
            label = f"unknown (was known {conf:.2f})"
        tag = "" if found else "  [no face detected - used whole image]"
        print(f"{os.path.basename(p):40s} {label:12s} {conf:6.2f}  "
              f"[{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f}]{tag}")


if __name__ == "__main__":
    main()
