"""
deploy_pi.py
------------
Live access control system for the Raspberry Pi.

Pipeline:
    camera frame -> face detection -> crop 224x224 -> TFLite inference
                 -> temporal smoothing -> decision -> GPIO output + log

GPIO behaviour (pins set in config.py):
    known      -> green LED steady
    unknown    -> red LED blinking
    intrusion  -> red LED steady + buzzer

Usage on the Pi:
    python3 deploy_pi.py --model models/mobilenetv2_int8.tflite

Test on a laptop without any GPIO hardware:
    python3 deploy_pi.py --model models/mobilenetv2_int8.tflite --no-gpio

Press q in the preview window (or Ctrl+C in headless mode) to quit.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

import config as C
from tflite_utils import TFLiteModel

MARGIN = 0.25


# --------------------------- GPIO ---------------------------

class Alerts:
    """Wraps the LEDs and buzzer. Falls back to console output when the
    GPIO library is unavailable, so the same script runs on a laptop."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.devices = None
        if not enabled:
            print("[i] GPIO disabled - alerts printed to the console")
            return
        try:
            # PWMOutputDevice drives the buzzer with a tone so it works with a
            # passive buzzer (needs an oscillating signal). It also works for an
            # active buzzer (a 50% duty cycle still powers it).
            from gpiozero import LED, PWMOutputDevice
            self.devices = {
                "green": LED(C.PIN_GREEN),
                "red": LED(C.PIN_RED),
                "buzzer": PWMOutputDevice(C.PIN_BUZZER, frequency=2000),
            }
            print(f"[i] GPIO ready - green:{C.PIN_GREEN} "
                  f"red:{C.PIN_RED} buzzer:{C.PIN_BUZZER}")
        except Exception as exc:
            print(f"[!] GPIO unavailable ({exc}); continuing without hardware")
            self.enabled = False

    def _all_off(self):
        if self.devices:
            self.devices["green"].off()
            self.devices["red"].off()
            self.devices["buzzer"].off()

    def set(self, state):
        if not self.devices:
            return
        self._all_off()
        if state == "known":
            self.devices["green"].on()
        elif state == "unknown":
            self.devices["red"].blink(on_time=0.3, off_time=0.3)
        elif state == "intrusion":
            self.devices["red"].on()
            self.devices["buzzer"].value = 0.5   # 50% duty -> passive buzzer beeps

    def close(self):
        if self.devices:
            self._all_off()
            for d in self.devices.values():
                d.close()


# --------------------------- vision ---------------------------

def to_square(img, size):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    img = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def crop_face(frame, box, size):
    x, y, w, h = box
    mx, my = int(w * MARGIN), int(h * MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    return to_square(frame[y0:y1, x0:x1], size)


def open_camera(index):
    """Works with a USB webcam and with the Pi camera module exposed
    through V4L2, which is the default on current Raspberry Pi OS."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("[!] Could not open the camera. Try --camera 1")
    return cap


def find_cascade():
    """Locate the frontal-face Haar cascade. cv2.data is missing on some
    OpenCV builds (e.g. the distro python3-opencv on Raspberry Pi OS), so fall
    back to the standard install locations."""
    name = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(cv2.data.haarcascades + name)
    candidates += [
        "/usr/share/opencv4/haarcascades/" + name,
        "/usr/share/opencv/haarcascades/" + name,
        "/usr/local/share/opencv4/haarcascades/" + name,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(f"[!] Could not find {name}. Install the OpenCV data files.")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=C.CONFIDENCE_THRESHOLD)
    ap.add_argument("--no-gpio", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="no preview window (for SSH sessions)")
    ap.add_argument("--log", default=os.path.join(C.LOG_DIR, "access_log.csv"))
    args = ap.parse_args()

    model = TFLiteModel(args.model, num_threads=args.threads)
    in_h, in_w = model.input_size
    print(f"[i] model {os.path.basename(args.model)} input {in_h}x{in_w}")

    detector = cv2.CascadeClassifier(find_cascade())
    cap = open_camera(args.camera)
    alerts = Alerts(enabled=not args.no_gpio)

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    new_log = not os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["timestamp", "decision", "confidence", "latency_ms"])

    recent = []                 # last few predictions, for smoothing
    current_state = None
    state_until = 0.0
    fps_times = []

    print("[i] running - press Ctrl+C to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[!] frame grab failed")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            decision, confidence = None, 0.0
            latency_ms = 0.0

            if len(faces) > 0:
                box = max(faces, key=lambda b: b[2] * b[3])
                face = crop_face(frame, box, in_h)
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

                t0 = time.perf_counter()
                probs = model.predict(rgb)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                label = C.CLASS_NAMES[idx]

                # A low-confidence face must never be accepted as known.
                if label == "known" and confidence < args.threshold:
                    label = "unknown"

                recent.append(label)
                if len(recent) > C.STABLE_FRAMES:
                    recent.pop(0)

                # act only when the last N frames agree
                if len(recent) == C.STABLE_FRAMES and len(set(recent)) == 1:
                    decision = recent[0]

                x, y, w, h = box
                colour = {"known": (0, 200, 0), "unknown": (0, 165, 255),
                          "intrusion": (0, 0, 255)}.get(label, (200, 200, 200))
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            else:
                recent.clear()

            now = time.time()
            if decision and (decision != current_state or now > state_until):
                current_state = decision
                state_until = now + C.ALERT_SECONDS
                alerts.set(decision)
                stamp = datetime.now().isoformat(timespec="seconds")
                writer.writerow([stamp, decision, round(confidence, 4),
                                 round(latency_ms, 2)])
                log_file.flush()
                print(f"[{stamp}] {decision:9s} conf={confidence:.3f} "
                      f"{latency_ms:.1f} ms")
            elif current_state and now > state_until and not decision:
                current_state = None
                alerts.set(None)

            fps_times.append(now)
            fps_times = [t for t in fps_times if now - t < 2.0]
            fps = len(fps_times) / 2.0

            if not args.headless:
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Intelligent Access Control", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[i] stopped by user")
    finally:
        alerts.close()
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"[i] access log written to {args.log}")


if __name__ == "__main__":
    main()
