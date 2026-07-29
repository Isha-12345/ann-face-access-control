"""
build_dataset.py
----------------
Builds the final three-class dataset (known / unknown / intrusion) for the
ANN security project from all downloaded sources.

Key properties:
  * Identity-aware splitting  -> all images of one person stay in one split,
                                 so the model cannot memorise faces.
  * Split-then-augment        -> augmentation is applied only to the training
                                 split, preventing data leakage.
  * Domain-gap reduction      -> web images receive JPEG compression, slight
                                 blur and colour jitter so they resemble
                                 camera output.
  * Face detection + crop     -> every image passes through the same
                                 224x224 pipeline as the deployment code.

Expected input layout (adjust paths with the arguments):

  sources/
    lfw/            <person>/img.jpg        (folder per identity)
    celeba/         img_00001.jpg ...       (flat folder)
    lfw_smfrd/      <person>/img.jpg        (folder per identity)
    rmfrd/          <person>/img.jpg        (folder per identity)
  dataset_raw/      <volunteer>/img.jpg     (from capture_faces.py)
  strangers_raw/    <stranger>/img.jpg      (same camera, NOT enrolled)

Output:

  dataset_final/
    train/{known,unknown,intrusion}/
    val/{known,unknown,intrusion}/
    test/{known,unknown,intrusion}/

Usage:
    python build_dataset.py --per-class 10000

Requires: opencv-python, numpy
"""

import argparse
import os
import random
import re

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

IMG_SIZE = 224
MARGIN = 0.25
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASSES = ["known", "unknown", "intrusion"]

_detector = None


def safe_tag(ident):
    """'vol:ramesh@train' -> 'ramesh' ; 'lfw:Aaron_Peirsol' -> 'Aaron_Peirsol'"""
    ident = ident.split(":", 1)[-1].split("@", 1)[0]
    return re.sub(r"[^A-Za-z0-9]+", "-", ident).strip("-") or "unk"


# --------------------------- face crop ---------------------------

def get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _detector


def to_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    img = cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def crop_face(img, allow_fallback=True):
    """Detect and crop the largest face. Falls back to a centre crop, which
    matters for masked images where the detector often fails."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = get_detector().detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        mx, my = int(w * MARGIN), int(h * MARGIN)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1 = min(img.shape[1], x + w + mx)
        y1 = min(img.shape[0], y + h + my)
        return to_square(img[y0:y1, x0:x1])
    if not allow_fallback:
        return None
    h, w = img.shape[:2]
    s = int(min(h, w) * 0.85)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return to_square(img[y0:y0 + s, x0:x0 + s])


# ------------------------ domain adaptation ------------------------

def web_to_camera_look(img):
    """Make a clean web image resemble a frame from a cheap camera."""
    if random.random() < 0.6:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.7:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.85, 1.15),
                                  beta=random.uniform(-20, 20))
    if random.random() < 0.5:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    q = random.randint(45, 80)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


# --------------------------- augmentation ---------------------------

def augment(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.7:
        ang = random.uniform(-12, 12)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.7:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.75, 1.25),
                                  beta=random.uniform(-25, 25))
    if random.random() < 0.4:
        h, w = img.shape[:2]
        z = random.uniform(0.85, 0.98)
        nh, nw = int(h * z), int(w * z)
        y0 = random.randint(0, h - nh)
        x0 = random.randint(0, w - nw)
        img = cv2.resize(img[y0:y0 + nh, x0:x0 + nw], (w, h))
    return img


# --------------------------- collecting ---------------------------

def collect_by_identity(root, tag):
    """For datasets with one folder per person. Returns {identity: [paths]}."""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        files = [os.path.join(pdir, f) for f in sorted(os.listdir(pdir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            out[f"{tag}:{person}"] = files
    return out


def collect_flat(root, tag):
    """For flat datasets such as CelebA. Each image is its own identity."""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    files = [f for f in sorted(os.listdir(root))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    for f in files:
        out[f"{tag}:{f}"] = [os.path.join(root, f)]
    return out


def split_identities(identities):
    """For UNKNOWN and INTRUSION: assign whole identities to splits, so the
    same person never appears in two splits. Targets the split ratios by
    image count rather than identity count."""
    ids = sorted(identities.keys(), key=lambda i: -len(identities[i]))
    total = sum(len(identities[i]) for i in ids)
    targets = {s: total * r for s, r in SPLITS.items()}
    assigned = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    # guarantee every split receives at least one identity
    for s in SPLITS:
        if ids:
            ident = ids.pop(0)
            assigned[s].append(ident)
            counts[s] += len(identities[ident])
    random.shuffle(ids)
    for ident in ids:
        s = max(SPLITS, key=lambda k: targets[k] - counts[k])
        assigned[s].append(ident)
        counts[s] += len(identities[ident])
    return assigned


def split_within_identity(identities):
    """For KNOWN: every enrolled person must appear in ALL splits, otherwise
    the model is asked to recognise a volunteer it never trained on. Images
    are therefore split inside each person's folder.

    Returns pseudo-identities of the form 'person@split'.
    """
    assigned = {s: [] for s in SPLITS}
    rebuilt = {}
    for ident, files in identities.items():
        files = list(files)
        random.shuffle(files)
        n = len(files)
        n_tr = int(n * SPLITS["train"])
        n_va = int(n * SPLITS["val"])
        parts = {
            "train": files[:n_tr],
            "val": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:],
        }
        for s, chunk in parts.items():
            if chunk:
                key = f"{ident}@{s}"
                rebuilt[key] = chunk
                assigned[s].append(key)
    return assigned, rebuilt


# ------------------------------ main ------------------------------

def process(paths_by_split, identities, cls, out_root, per_class,
            web_source, aug_train, stats):
    quota = {s: int(per_class * r) for s, r in SPLITS.items()}
    for split, idents in paths_by_split.items():
        dst = os.path.join(out_root, split, cls)
        os.makedirs(dst, exist_ok=True)
        files = [(i, f) for i in idents for f in identities[i]]
        random.shuffle(files)

        n_needed = quota[split]
        # if augmenting, fewer originals are required
        n_take = n_needed if not (aug_train and split == "train") \
            else max(1, n_needed // (1 + aug_train))

        saved = 0
        for ident, src in files:
            if saved >= n_needed:
                break
            img = cv2.imread(src)
            if img is None:
                continue
            face = crop_face(img)
            if face is None:
                continue
            if web_source:
                face = web_to_camera_look(face)

            # identity is preserved in the filename so that Task 2
            # (embedding model) can recover which volunteer each image
            # belongs to. Format:  <class>__<identity>__<split>_<n>.jpg
            tag = safe_tag(ident)
            base = f"{cls}__{tag}__{split}_{saved:06d}"
            cv2.imwrite(os.path.join(dst, base + ".jpg"), face,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1

            if aug_train and split == "train":
                for k in range(aug_train):
                    if saved >= n_needed:
                        break
                    cv2.imwrite(os.path.join(dst, f"{base}_aug{k}.jpg"),
                                augment(face), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    saved += 1
            if saved >= n_take and not aug_train:
                continue
        stats[cls][split] = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="sources/lfw")
    ap.add_argument("--celeba", default="sources/celeba")
    ap.add_argument("--smfrd", default="sources/lfw_smfrd")
    ap.add_argument("--rmfrd", default="sources/rmfrd")
    ap.add_argument("--known", default="dataset_raw")
    ap.add_argument("--strangers", default="strangers_raw")
    ap.add_argument("--out", default="dataset_final")
    ap.add_argument("--per-class", type=int, default=10000)
    args = ap.parse_args()

    stats = {c: {} for c in CLASSES}

    # ---------- KNOWN ----------
    known = collect_by_identity(args.known, "vol")
    if not known:
        raise SystemExit(f"[!] No volunteer images found in {args.known}")
    print(f"[i] known: {len(known)} identities, "
          f"{sum(len(v) for v in known.values())} images")
    known_splits, known_ids = split_within_identity(known)
    process(known_splits, known_ids, "known", args.out,
            args.per_class, web_source=False, aug_train=4, stats=stats)

    # ---------- UNKNOWN ----------
    unknown = {}
    unknown.update(collect_by_identity(args.lfw, "lfw"))
    unknown.update(collect_flat(args.celeba, "celeba"))
    strangers = collect_by_identity(args.strangers, "stranger")
    unknown.update(strangers)
    print(f"[i] unknown: {len(unknown)} identities "
          f"({len(strangers)} same-camera strangers)")
    process(split_identities(unknown), unknown, "unknown", args.out,
            args.per_class, web_source=True, aug_train=0, stats=stats)

    # ---------- INTRUSION ----------
    intrusion = {}
    intrusion.update(collect_by_identity(args.smfrd, "smfrd"))
    intrusion.update(collect_by_identity(args.rmfrd, "rmfrd"))
    print(f"[i] intrusion: {len(intrusion)} identities")
    process(split_identities(intrusion), intrusion, "intrusion", args.out,
            args.per_class, web_source=True, aug_train=0, stats=stats)

    print("\n[i] Final dataset:")
    for c in CLASSES:
        row = " | ".join(f"{s}: {stats[c].get(s, 0)}" for s in SPLITS)
        print(f"    {c:10s} {row}")
    print(f"\n[i] Written to {args.out}/")
    print("[i] Reminder: keep masked photos of your KNOWN volunteers as a "
          "separate hard-negative test set. Do not train on them.")


if __name__ == "__main__":
    main()
