"""
train_classifier.py
-------------------
Task 1: multi-class classification (known / unknown / intrusion).

Trains one of three architectures:
  * mobilenetv2  -- transfer learning, two-stage (frozen then fine-tuned)
  * mobilenetv3  -- transfer learning, two-stage
  * baseline     -- small CNN trained from scratch (comparison model)

Usage:
    python train_classifier.py --model mobilenetv2
    python train_classifier.py --model mobilenetv3
    python train_classifier.py --model baseline

Outputs (in models/ and results/):
    models/<name>.keras            trained model
    results/<name>_history.json    per-epoch accuracy and loss
    results/<name>_curves.png      training curves
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config as C

tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)


# ------------------------------ data ------------------------------

def load_split(split, shuffle):
    path = os.path.join(C.DATA_DIR, split)
    if not os.path.isdir(path):
        raise SystemExit(f"[!] Missing folder: {path}. Run build_dataset.py first.")
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=C.CLASS_NAMES,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        seed=C.SEED,
    )
    return ds


def compute_class_weights(train_ds):
    """Counts images per class from the directory, not by iterating batches."""
    counts = []
    for cls in C.CLASS_NAMES:
        d = os.path.join(C.DATA_DIR, "train", cls)
        counts.append(len(os.listdir(d)) if os.path.isdir(d) else 0)
    counts = np.array(counts, dtype=np.float64)
    total = counts.sum()
    weights = total / (len(counts) * np.maximum(counts, 1))
    print("[i] class counts:", dict(zip(C.CLASS_NAMES, counts.astype(int))))
    print("[i] class weights:", dict(zip(C.CLASS_NAMES, weights.round(3))))
    return {i: float(w) for i, w in enumerate(weights)}


# ------------------------------ models ------------------------------

def build_transfer(name):
    """MobileNetV2 and MobileNetV3-Small expect different input ranges, so
    each gets its own preprocessing layer. Inputs are raw 0-255 images."""
    inputs = tf.keras.Input(shape=C.IMG_SHAPE)

    if name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=C.IMG_SHAPE, include_top=False, weights="imagenet")
    elif name == "mobilenetv3":
        # MobileNetV3 in Keras performs its own rescaling from 0-255
        x = inputs
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=C.IMG_SHAPE, include_top=False, weights="imagenet",
            include_preprocessing=True)
    else:
        raise ValueError(name)

    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(C.DROPOUT)(x)
    outputs = tf.keras.layers.Dense(len(C.CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    return model, backbone


def build_baseline():
    """Small CNN trained from scratch. Included to quantify how much
    transfer learning actually contributes."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=C.IMG_SHAPE),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(C.DROPOUT),
        tf.keras.layers.Dense(len(C.CLASS_NAMES), activation="softmax"),
    ], name="baseline")
    return model, None


# ------------------------------ plots ------------------------------

def plot_history(hist, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="validation")
    axes[0].set_title(f"{name} - accuracy")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="validation")
    axes[1].set_title(f"{name} - loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(C.RESULT_DIR, f"{name}_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[i] saved {out}")


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mobilenetv2", "mobilenetv3", "baseline"])
    ap.add_argument("--epochs-frozen", type=int, default=C.EPOCHS_FROZEN)
    ap.add_argument("--epochs-finetune", type=int, default=C.EPOCHS_FINETUNE)
    args = ap.parse_args()

    C.configure_cpu_threads()
    print(f"[i] TensorFlow {tf.__version__} | CPU threads: "
          f"intra={C.INTRA_OP_THREADS} inter={C.INTER_OP_THREADS}")
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    # the validation set is re-read every epoch and is never augmented, so caching
    # it in memory skips repeated JPEG decoding for a free per-epoch speed-up
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    class_weight = compute_class_weights(train_ds)

    if args.model == "baseline":
        model, backbone = build_baseline()
    else:
        model, backbone = build_transfer(args.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(C.LR_FROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt = os.path.join(C.MODEL_DIR, f"{args.model}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, verbose=1),
    ]

    # ---- stage 1 ----
    print("\n===== STAGE 1: frozen backbone =====")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.epochs_frozen,
                   class_weight=class_weight, callbacks=callbacks)
    history = {k: list(map(float, v)) for k, v in h1.history.items()}

    # ---- stage 2 ----
    if backbone is not None and args.epochs_finetune > 0:
        print("\n===== STAGE 2: fine-tuning =====")
        backbone.trainable = True
        for layer in backbone.layers[:-C.UNFREEZE_LAYERS]:
            layer.trainable = False
        # BatchNorm layers stay frozen: unfreezing them with small batches
        # destabilises transfer learning.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        trainable = sum(1 for l in backbone.layers if l.trainable)
        print(f"[i] unfrozen backbone layers: {trainable}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(C.LR_FINETUNE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.epochs_finetune,
                       class_weight=class_weight, callbacks=callbacks)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(map(float, v))

    model.save(ckpt)
    print(f"[i] saved {ckpt}")

    hist_path = os.path.join(C.RESULT_DIR, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, args.model)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[i] {args.model}: validation accuracy = {acc:.4f}, loss = {loss:.4f}")
    print("[i] Next: python evaluate.py --model", args.model)


if __name__ == "__main__":
    main()
