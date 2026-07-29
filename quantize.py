"""
quantize.py
-----------
Converts a trained Keras model to TensorFlow Lite in two forms and measures
the accuracy / size trade-off. These numbers are the core of the edge
optimisation section of the report.

  1. float32  -- straight conversion, no compression
  2. INT8     -- full integer quantisation using a representative dataset

Usage:
    python quantize.py --model mobilenetv2

Outputs:
    models/<name>_float32.tflite
    models/<name>_int8.tflite
    results/<name>_quantisation.json
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import tensorflow as tf

import config as C
from tflite_utils import TFLiteModel


def representative_dataset(n=200):
    """Feeds real training images so the converter can calibrate the
    activation ranges for each layer."""
    paths = []
    for cls in C.CLASS_NAMES:
        paths += glob.glob(os.path.join(C.DATA_DIR, "train", cls, "*.jpg"))
    rng = np.random.default_rng(C.SEED)
    rng.shuffle(paths)

    def gen():
        for p in paths[:n]:
            img = tf.keras.utils.load_img(p, target_size=(C.IMG_SIZE, C.IMG_SIZE))
            arr = tf.keras.utils.img_to_array(img).astype(np.float32)
            yield [np.expand_dims(arr, 0)]
    return gen


def convert(model, int8, rep_gen=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    return converter.convert()


def load_test_arrays(max_per_class=400):
    xs, ys = [], []
    for idx, cls in enumerate(C.CLASS_NAMES):
        paths = sorted(glob.glob(os.path.join(C.DATA_DIR, "test", cls, "*.jpg")))
        for p in paths[:max_per_class]:
            img = tf.keras.utils.load_img(p, target_size=(C.IMG_SIZE, C.IMG_SIZE))
            xs.append(tf.keras.utils.img_to_array(img).astype(np.float32))
            ys.append(idx)
    return np.array(xs), np.array(ys)


def tflite_accuracy(path, x, y):
    model = TFLiteModel(path, num_threads=4)
    correct, times = 0, []
    for i in range(len(x)):
        t0 = time.perf_counter()
        prob = model.predict(x[i])
        times.append((time.perf_counter() - t0) * 1000)
        correct += int(np.argmax(prob) == y[i])
    return correct / max(len(x), 1), float(np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mobilenetv2", "mobilenetv3", "baseline"])
    args = ap.parse_args()

    keras_path = os.path.join(C.MODEL_DIR, f"{args.model}.keras")
    if not os.path.exists(keras_path):
        raise SystemExit(f"[!] Model not found: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    keras_size = os.path.getsize(keras_path) / 1e6
    print(f"[i] original Keras model: {keras_size:.2f} MB")

    f32_path = os.path.join(C.MODEL_DIR, f"{args.model}_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(convert(model, int8=False))
    f32_size = os.path.getsize(f32_path) / 1e6
    print(f"[i] float32 TFLite: {f32_size:.2f} MB -> {f32_path}")

    int8_path = os.path.join(C.MODEL_DIR, f"{args.model}_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(convert(model, int8=True, rep_gen=representative_dataset()))
    int8_size = os.path.getsize(int8_path) / 1e6
    print(f"[i] INT8 TFLite:    {int8_size:.2f} MB -> {int8_path}")

    print("\n[i] measuring accuracy on the test split ...")
    x, y = load_test_arrays()
    if len(x) == 0:
        raise SystemExit("[!] No test images found.")
    keras_acc = float((model.predict(x, batch_size=C.BATCH_SIZE,
                                     verbose=0).argmax(1) == y).mean())
    f32_acc, f32_ms = tflite_accuracy(f32_path, x, y)
    int8_acc, int8_ms = tflite_accuracy(int8_path, x, y)

    report = {
        "model": args.model,
        "n_test_images": int(len(x)),
        "keras": {"size_mb": round(keras_size, 3), "accuracy": round(keras_acc, 4)},
        "tflite_float32": {"size_mb": round(f32_size, 3),
                           "accuracy": round(f32_acc, 4),
                           "latency_ms_this_machine": round(f32_ms, 2)},
        "tflite_int8": {"size_mb": round(int8_size, 3),
                        "accuracy": round(int8_acc, 4),
                        "latency_ms_this_machine": round(int8_ms, 2)},
        "size_reduction_vs_float32": f"{(1 - int8_size / max(f32_size, 1e-9)) * 100:.1f}%",
        "accuracy_change_int8": round(int8_acc - f32_acc, 4),
    }
    out = os.path.join(C.RESULT_DIR, f"{args.model}_quantisation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== quantisation summary =====")
    print(f"  Keras          {keras_size:7.2f} MB   accuracy {keras_acc:.4f}")
    print(f"  TFLite float32 {f32_size:7.2f} MB   accuracy {f32_acc:.4f}   "
          f"{f32_ms:.1f} ms/image")
    print(f"  TFLite INT8    {int8_size:7.2f} MB   accuracy {int8_acc:.4f}   "
          f"{int8_ms:.1f} ms/image")
    print(f"  size reduction: {report['size_reduction_vs_float32']}, "
          f"accuracy change: {report['accuracy_change_int8']:+.4f}")
    print(f"\n[i] saved {out}")
    print("[i] Copy the INT8 model to the Raspberry Pi and run benchmark_pi.py")


if __name__ == "__main__":
    main()
