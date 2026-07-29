"""Independent test: run the CURRENT trained model on the newly-added friend
photos it never saw (files NOT matching <friend>_NNNN.jpg, i.e. added after the
dataset was built). Reports per-friend recognition and the overall figure."""
import os, re
import numpy as np
import cv2
from PIL import Image, ImageOps
import pillow_heif
import tensorflow as tf
import config as C

pillow_heif.register_heif_opener()
MARGIN = 0.25
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model(os.path.join(C.MODEL_DIR, "mobilenetv2.keras"))

def to_square(img, s=224):
    h, w = img.shape[:2]; m = max(h, w)
    t, l = (m - h)//2, (m - w)//2
    img = cv2.copyMakeBorder(img, t, m-h-t, l, m-w-l, cv2.BORDER_CONSTANT, value=(0,0,0))
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)

def load_bgr(path):
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def crop(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = det.detectMultiScale(g, 1.1, 5, minSize=(50,50))
    if len(f):
        x,y,w,h = max(f, key=lambda b: b[2]*b[3])
        mx,my = int(w*MARGIN), int(h*MARGIN)
        return to_square(img[max(0,y-my):y+h+my, max(0,x-mx):x+w+mx])
    return to_square(img)

ROOT = "dataset_raw"
print(f"{'friend':11s} {'new photos':>10s} {'->known':>8s} {'unknown':>8s} {'intrusion':>9s}  recog%")
grand_tot = grand_known = 0
for friend in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, friend)
    if not os.path.isdir(d): continue
    pat = re.compile(rf"^{re.escape(friend)}_[0-9]+\.jpg$")
    new = [f for f in os.listdir(d) if not pat.match(f)
           and f.lower().endswith((".jpg",".jpeg",".png",".heic",".heif",".webp"))]
    if not new: continue
    cnt = {"known":0,"unknown":0,"intrusion":0}
    for f in new:
        img = load_bgr(os.path.join(d, f))
        if img is None: continue
        rgb = cv2.cvtColor(crop(img), cv2.COLOR_BGR2RGB).astype(np.float32)
        p = model.predict(np.expand_dims(rgb,0), verbose=0)[0]
        lab = C.CLASS_NAMES[int(np.argmax(p))]
        if lab == "known" and float(p.max()) < C.CONFIDENCE_THRESHOLD: lab = "unknown"
        cnt[lab]+=1
    tot = sum(cnt.values())
    recog = 100*cnt["known"]/tot if tot else 0
    print(f"{friend:11s} {tot:10d} {cnt['known']:8d} {cnt['unknown']:8d} {cnt['intrusion']:9d}  {recog:5.1f}%")
    grand_tot += tot; grand_known += cnt["known"]
print(f"\nOVERALL: {grand_known} of {grand_tot} new photos recognised as their own identity "
      f"= {100*grand_known/max(grand_tot,1):.1f}%")
