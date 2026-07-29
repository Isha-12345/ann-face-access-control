"""
retune_embedding.py
-------------------
Re-threshold the open-set embedding model (Task 2) WITHOUT retraining.

Loads the saved encoder + centroids, recomputes cosine distances on the test
split, then sweeps every threshold to expose the FAR/FRR trade-off. Reports
several operating points (original, equal-error, target-FRR) and breaks FAR
down into unmasked-stranger vs masked-intruder, which is where the model
struggles. Saves a curve figure and an updated metrics table.

Usage:  python retune_embedding.py           # keep chosen threshold in a new file
        python retune_embedding.py --apply    # also bake it into centroids.npz
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import config as C

TARGET_FRR = 0.10          # design goal: reject at most ~10% of real friends


def load_split(split, cls):
    paths = sorted(glob.glob(os.path.join(C.DATA_DIR, split, cls, "*.jpg")))
    x = np.zeros((len(paths), C.IMG_SIZE, C.IMG_SIZE, 3), np.float32)
    for i, p in enumerate(paths):
        x[i] = tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(p, target_size=(C.IMG_SIZE, C.IMG_SIZE)))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the recommended threshold into centroids.npz")
    args = ap.parse_args()

    d = np.load(os.path.join(C.MODEL_DIR, "centroids.npz"), allow_pickle=True)
    centroids = d["centroids"]
    old_thr = float(d["threshold"])
    # The saved embedding.keras uses a Lambda(l2_normalize) whose inline lambda
    # loses its `tf` reference when deserialized in a fresh process. Rebuild the
    # encoder from the training code (identical architecture, real function) and
    # restore the trained weights from inside the .keras archive.
    import tempfile
    import zipfile
    from train_embedding import build_embedding_model
    _, enc, _ = build_embedding_model(len(d["identities"]))
    tmpd = tempfile.mkdtemp()
    with zipfile.ZipFile(os.path.join(C.MODEL_DIR, "embedding.keras")) as z:
        wname = next(n for n in z.namelist() if n.endswith(".weights.h5"))
        z.extract(wname, tmpd)
    enc.load_weights(os.path.join(tmpd, wname))

    def dist(x):
        emb = enc.predict(x, batch_size=C.BATCH_SIZE, verbose=0)
        return 1.0 - (emb @ centroids.T).max(axis=1)

    d_known = dist(load_split("test", "known"))
    d_unknown = dist(load_split("test", "unknown"))
    d_intrusion = dist(load_split("test", "intrusion"))
    d_novel = np.concatenate([d_unknown, d_intrusion])

    # threshold-independent separation quality (known=small dist, novel=large)
    auc = float(roc_auc_score(
        np.r_[np.zeros(len(d_known)), np.ones(len(d_novel))],
        np.r_[d_known, d_novel]))

    def at(t):
        return {
            "threshold": round(float(t), 4),
            "FRR": round(float((d_known > t).mean()), 4),
            "FAR": round(float((d_novel <= t).mean()), 4),
            "FAR_unknown": round(float((d_unknown <= t).mean()), 4),
            "FAR_intrusion": round(float((d_intrusion <= t).mean()), 4),
            "accuracy": round(float(((d_known <= t).sum() + (d_novel > t).sum())
                                    / (len(d_known) + len(d_novel))), 4),
        }

    ts = np.linspace(0, 1, 1001)
    frrs = np.array([(d_known > t).mean() for t in ts])
    fars = np.array([(d_novel <= t).mean() for t in ts])

    eer_t = ts[int(np.argmin(np.abs(frrs - fars)))]
    bal_t = ts[int(np.argmin(frrs + fars))]
    frr_ok = np.where(frrs <= TARGET_FRR)[0]
    tgt_t = ts[frr_ok[0]] if len(frr_ok) else ts[-1]

    points = {
        "original": at(old_thr),
        f"target_FRR_{int(TARGET_FRR*100)}pct": at(tgt_t),
        "equal_error": at(eer_t),
        "balanced_min(FAR+FRR)": at(bal_t),
    }

    print(f"\n[i] separation AUC (known vs novel): {auc:.4f}  (1.0 = perfect)")
    print(f"[i] mean distance  known={d_known.mean():.3f}  "
          f"unknown={d_unknown.mean():.3f}  intrusion={d_intrusion.mean():.3f}\n")
    hdr = f"{'operating point':22s} {'thr':>6s} {'FRR':>7s} {'FAR':>7s} " \
          f"{'FAR_unk':>8s} {'FAR_intr':>9s} {'acc':>7s}"
    print(hdr); print("-" * len(hdr))
    for name, m in points.items():
        print(f"{name:22s} {m['threshold']:6.3f} {m['FRR']*100:6.1f}% "
              f"{m['FAR']*100:6.1f}% {m['FAR_unknown']*100:7.1f}% "
              f"{m['FAR_intrusion']*100:8.1f}% {m['accuracy']*100:6.1f}%")

    # Equal-error is the standard biometric operating point to report. The
    # target-FRR point is deliberately NOT used: it accepts ~99% of masked
    # intruders, which defeats the security purpose.
    recommended = points["equal_error"]
    report = {"auc_known_vs_novel": round(auc, 4),
              "mean_distance": {"known": round(float(d_known.mean()), 4),
                                "unknown": round(float(d_unknown.mean()), 4),
                                "intrusion": round(float(d_intrusion.mean()), 4)},
              "operating_points": points,
              "recommended": recommended}
    out = os.path.join(C.RESULT_DIR, "embedding_retuned_metrics.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[i] saved {out}")

    # ---- figure: trade-off curve + distance distributions ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ts, frrs * 100, label="FRR (friends rejected)")
    ax[0].plot(ts, fars * 100, label="FAR (intruders accepted)")
    for name, t, c in [("original", old_thr, "gray"),
                       ("target 10% FRR", tgt_t, "green"),
                       ("equal-error", eer_t, "orange")]:
        ax[0].axvline(t, color=c, ls="--", alpha=0.8, label=f"{name} = {t:.3f}")
    ax[0].set_xlabel("threshold (cosine distance)"); ax[0].set_ylabel("error rate (%)")
    ax[0].set_title("FAR / FRR vs threshold"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_xlim(0, 0.6)

    bins = np.linspace(0, 0.6, 50)
    ax[1].hist(d_known, bins=bins, alpha=0.6, label="known (friends)")
    ax[1].hist(d_unknown, bins=bins, alpha=0.6, label="unknown (strangers)")
    ax[1].hist(d_intrusion, bins=bins, alpha=0.6, label="intrusion (masked)")
    ax[1].axvline(old_thr, color="gray", ls="--", label=f"original {old_thr:.3f}")
    ax[1].axvline(tgt_t, color="green", ls="--", label=f"retuned {tgt_t:.3f}")
    ax[1].set_xlabel("cosine distance to nearest enrolled friend")
    ax[1].set_ylabel("# test images"); ax[1].set_title("Distance distributions")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    figout = os.path.join(C.RESULT_DIR, "embedding_threshold_curve.png")
    fig.savefig(figout, dpi=150); plt.close(fig)
    print(f"[i] saved {figout}")

    if args.apply:
        np.savez(os.path.join(C.MODEL_DIR, "centroids.npz"),
                 centroids=centroids, identities=d["identities"],
                 threshold=recommended["threshold"])
        print(f"[i] centroids.npz threshold updated -> {recommended['threshold']}")
    else:
        print("[i] (re-run with --apply to bake the recommended threshold into centroids.npz)")


if __name__ == "__main__":
    main()
