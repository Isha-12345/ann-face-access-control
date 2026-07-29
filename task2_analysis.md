# Task 2 — Open-Set Anomaly Detection: Analysis and Threshold Selection

## Approach

The Task 1 classifier is closed-set: it forces every face into one of three
classes, even one it has never seen. For access control this is unsafe, because
the system meets mostly unseen people. Task 2 therefore reframes the problem as
open-set verification. A MobileNetV2 backbone is trained on the six enrolled
volunteers only, producing a 128-dimensional L2-normalised embedding that
separates the individual identities. One centroid vector is stored per enrolled
person. At test time a face is embedded and its cosine distance to the nearest
centroid is measured; if that distance exceeds a threshold the face is treated
as novel and access is denied. The model never trains on unknown or intrusion
images, so it generalises to people who were never in the dataset.

Two errors matter:

- **FAR** (False Acceptance Rate) — a non-enrolled person accepted as known.
  This is the security-critical error.
- **FRR** (False Rejection Rate) — an enrolled person wrongly rejected.

## Results

Measured on the held-out test split (known 294, unknown 375, intrusion 375):

| Operating point | Threshold | FRR | FAR | FAR (unmasked) | FAR (masked) | Accuracy |
|---|---|---|---|---|---|---|
| Security-first | 0.054 | 56.5% | 3.3% | 0.3% | 6.4% | 81.7% |
| Target 10% FRR | 0.286 | 9.9% | 67.2% | 35.7% | 98.7% | 48.9% |
| **Equal-error (reported)** | **0.116** | **31.6%** | **31.7%** | **5.3%** | **58.1%** | **68.3%** |
| Balanced (min FAR+FRR) | 0.074 | 45.2% | 10.5% | 1.6% | 19.5% | 79.7% |

Separation quality (threshold-independent) is AUC = 0.77. Mean cosine distance
to the nearest enrolled centroid is 0.111 for enrolled faces, 0.343 for
unmasked strangers, and 0.115 for masked intruders.

*(Figures: `results/embedding_distances.png` for the distance histograms and
`results/embedding_threshold_curve.png` for the FAR/FRR trade-off curve.)*

## Interpretation

The headline figure is the near-identical mean distance of enrolled faces
(0.111) and masked intruders (0.115). In the embedding space these two groups
occupy the same region, so no threshold can separate them: the "Target 10% FRR"
row shows that admitting 90% of enrolled users would simultaneously admit 98.7%
of masked intruders. Threshold tuning does not fix this; it only trades FRR
against masked-FAR.

Splitting FAR by occlusion explains why. Against **unmasked** strangers the
model performs well — FAR stays between 0.3% and 5% across sensible thresholds,
confirming the embedding genuinely distinguishes the enrolled volunteers from
ordinary strangers. Against **masked** faces it fails at every usable threshold,
because a mask removes the identifying facial features the embedding depends on.
This is a known limitation of face embeddings under occlusion rather than a
defect of this particular model.

This also explains the Task 1 result. The supervised classifier reaches ~99%
and detects intrusions reliably because it learned "masked face → intrusion" as
a direct visual pattern, instead of reasoning about identity distance. The two
approaches are therefore complementary, not competing.

## Chosen operating point and recommendation

The equal-error threshold (0.116, where FRR ≈ FAR ≈ 32%) is adopted as the
reported operating point, following standard biometric practice. It is baked
into `models/centroids.npz`. The very low security-first threshold (0.054) was
rejected because its 56% FRR makes the system unusable for legitimate users, and
the 10% FRR point was rejected because it defeats the security purpose entirely.

For deployment the recommended design is a **two-stage cascade**: the Task 1
classifier acts as the primary detector (high accuracy, robust to masks), and
the open-set embedding acts as a secondary identity check for unmasked faces,
where its low false-acceptance rate is most valuable. This combines the
classifier's strength on occlusion with the embedding's ability to reject
previously unseen, unmasked individuals.
