# How the research was carried out

## 1. Problem
Signature and supervised IDS models only recognise attacks seen in training. Zero-day attacks are, by definition, unseen. This study asks which *lightweight, CPU-only* detectors best flag unseen attack categories, and whether the answer holds across datasets.

## 2. Protocol: leave-attack-category-out
For every attack category C in a dataset:
1. Withhold all samples of C from training (C plays the zero-day attack).
2. Split benign traffic and the remaining (known) attack categories 70/30 into train/test (`random_state=42`).
3. Train each detector on the training portion only (benign, plus known attacks where the method uses labels).
4. Score a test set made of held-out benign, held-out known attacks and all of C.
5. Record metrics and deployment cost, then repeat for the next category.

## 3. Datasets
| Dataset | Rows used | Categories (held out one at a time) |
|---|---|---|
| NSL-KDD | 148,517 (Train+ and Test+) | DoS, Probe, R2L, U2R |
| CICIoT2023 | 744,843 (18% stratified per-partition sample of ~6M) | DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai |

CICIoT2023's 34 fine-grained labels are mapped to 7 categories plus Benign (mapping in `src/data_prep.py`). Features are standardised with `StandardScaler`.

## 4. Detectors (scikit-learn and NumPy only, no GPU)
Isolation Forest; One-Class SVM (training subsample capped at 4,000); PCA-Reconstruction (95% variance); MLP-Autoencoder (32-8-32 bottleneck); Prototype-Contrastive (per-class centroids with Ledoit-Wolf shared covariance, score = Mahalanobis distance to nearest prototype).

## 5. Metrics
- AUROC on known attacks and on the unknown category (each against benign)
- Unknown Detection Rate (UDR) at a threshold fixed for 5% benign false-positive rate
- Training time, per-sample inference latency (ms), serialised model size (KB)

## 6. Key findings
Mean AUROC on the unknown category:

| Detector | NSL-KDD | CICIoT2023 |
|---|---|---|
| Isolation Forest | 0.891 | 0.799 |
| One-Class SVM | 0.864 | 0.697 |
| PCA-Reconstruction | 0.933 | 0.796 |
| MLP-Autoencoder | 0.928 | 0.784 |
| Prototype-Contrastive | 0.951 | 0.419 |

Rankings do not transfer. Prototype-Contrastive wins on NSL-KDD but is worst on CICIoT2023, where its shared-covariance assumption breaks under extreme class imbalance (e.g. DDoS 542,686 vs BruteForce 203 rows). Isolation Forest and PCA-Reconstruction are robust on both; PCA-Reconstruction is also the cheapest (about 5-22 KB, sub-millisecond inference).

## 7. Engineering notes
- A shrinkage parameter in Prototype-Contrastive was originally defined but unused; it now uses `sklearn.covariance.LedoitWolf`.
- CICIoT2023 is sampled by chunked streaming so the full 6M rows never sit in memory.
- Hardware: laptop-class CPU, about 4 GB RAM; large categories (DDoS) were run one detector at a time.

## 8. Limitations
Single seed (no variance estimates), one threshold (5% FPR), two datasets, sampled CICIoT2023. See the paper's future-work section.
