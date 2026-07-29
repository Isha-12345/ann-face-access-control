"""
convert_to_jpg.py
-----------------
One-off helper: convert HEIC/HEIF/WEBP photos (which OpenCV cannot read) into
JPG so build_dataset.py picks them up. Originals are left untouched; a new
.jpg is written alongside each. EXIF orientation is applied so portrait phone
photos come out upright (build_dataset uses cv2.imread, which ignores EXIF).

Usage:
    python convert_to_jpg.py dataset_raw strangers_raw hard_negatives
"""
import os
import sys
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()  # lets PIL open .heic / .heif

EXTS = (".heic", ".heif", ".webp")


def convert_tree(root):
    converted, skipped, failed = 0, 0, 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            src = os.path.join(dirpath, f)
            dst = os.path.join(dirpath, os.path.splitext(f)[0] + ".jpg")
            if os.path.exists(dst):
                skipped += 1
                continue
            try:
                img = Image.open(src)
                img = ImageOps.exif_transpose(img)      # honour rotation
                img.convert("RGB").save(dst, "JPEG", quality=95)
                converted += 1
            except Exception as exc:
                print(f"  [!] failed {src}: {exc}")
                failed += 1
    return converted, skipped, failed


if __name__ == "__main__":
    roots = sys.argv[1:] or ["dataset_raw"]
    total_c = total_s = total_f = 0
    for r in roots:
        if not os.path.isdir(r):
            print(f"[i] skip (not found): {r}")
            continue
        c, s, fa = convert_tree(r)
        print(f"[i] {r}: converted {c}, already-existed {s}, failed {fa}")
        total_c += c; total_s += s; total_f += fa
    print(f"\n[i] TOTAL converted {total_c}, skipped {total_s}, failed {total_f}")
