"""
capture_faces.py
----------------
Captures face images from a webcam / Raspberry Pi camera for the "known"
class of the ANN security project.

Usage:
    python capture_faces.py --name ramesh --out dataset_raw --target 250

Controls while running:
    SPACE : toggle auto-capture on/off
    q     : quit

Output:
    dataset_raw/<name>/<name>_0001.jpg ... (cropped face images, 224x224)

Requires: opencv-python  (pip install opencv-python)
Works on laptop webcams and on Raspberry Pi (with a USB cam or the Pi
camera exposed through V4L2, which is the default on Raspberry Pi OS
Bullseye/Bookworm).
"""

import argparse
import os
import time

import cv2

MARGIN = 0.25          # extra margin around the detected face box
IMG_SIZE = 224         # output crop size (matches MobileNet input)
CAPTURE_INTERVAL = 0.4 # seconds between saved frames (avoids near-duplicates)
MIN_FACE = 80          # ignore tiny faces (person too far / false positive)
BLUR_THRESHOLD = 60.0  # variance-of-Laplacian below this = too blurry, skip


def crop_face(frame, box):
    """Crop the face with a margin, padded to square, resized to IMG_SIZE."""
    x, y, w, h = box
    mx, my = int(w * MARGIN), int(h * MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    face = frame[y0:y1, x0:x1]

    # pad to square so resizing does not distort the face
    fh, fw = face.shape[:2]
    size = max(fh, fw)
    top = (size - fh) // 2
    left = (size - fw) // 2
    face = cv2.copyMakeBorder(
        face, top, size - fh - top, left, size - fw - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def is_sharp(gray_crop):
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var() >= BLUR_THRESHOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="person's name (folder name)")
    ap.add_argument("--out", default="dataset_raw", help="output root folder")
    ap.add_argument("--target", type=int, default=250, help="images to collect")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.name.lower())
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    count = existing
    print(f"[i] Saving to {out_dir} (resuming at {existing} images)")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("[!] Could not open camera. Try --camera 1")

    capturing = False
    last_saved = 0.0

    while count < args.target:
        ok, frame = cap.read()
        if not ok:
            print("[!] Frame grab failed"); break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(MIN_FACE, MIN_FACE),
        )

        display = frame.copy()
        # only act when exactly one face is visible -> clean labels
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if capturing and time.time() - last_saved >= CAPTURE_INTERVAL:
                crop = crop_face(frame, (x, y, w, h))
                if is_sharp(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
                    count += 1
                    path = os.path.join(out_dir, f"{args.name.lower()}_{count:04d}.jpg")
                    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    last_saved = time.time()

        status = f"{args.name}: {count}/{args.target}  " + \
                 ("[CAPTURING - SPACE to pause]" if capturing else "[PAUSED - SPACE to start]")
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if capturing else (0, 0, 255), 2)
        cv2.imshow("Known-face capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            capturing = not capturing
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[i] Done. {count} images in {out_dir}")


if __name__ == "__main__":
    main()
