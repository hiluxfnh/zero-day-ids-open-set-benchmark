"""
run_experiment.py
Leave-category-out benchmark driver.

Usage (NSL-KDD dry run, used to validate the pipeline before CICIoT2023
is available):
    python run_experiment.py --dataset nslkdd \
        --nslkdd-train data/nslkdd/KDDTrain+.txt \
        --nslkdd-test  data/nslkdd/KDDTest+.txt \
        --out results_nslkdd_dryrun.csv

Usage (CICIoT2023, once downloaded and placed under a folder of CSVs):
    python run_experiment.py --dataset ciciot2023 \
        --ciciot-dir data/ciciot2023/ \
        --sample-frac 0.05 \
        --out results_ciciot2023.csv

For each attack CATEGORY present, the category is held out entirely from
training (simulating a zero-day attack never seen before), every detector
is fit on benign (+ remaining known attack categories, where the method
uses them) and scored on a held-out test split containing benign, known
attacks, and the unknown category. Results are written long-form to a CSV:
one row per (held_out_category, detector, metric-set).

Use --categories to run only a subset of held-out categories at a time
(useful for chunking long runs) and --append to accumulate into an
existing results CSV across multiple invocations.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_prep import load_nslkdd, featurize_nslkdd, load_ciciot2023_sampled, featurize_ciciot2023, RANDOM_STATE
from open_set_methods import ALL_DETECTORS, profile_detector
from evaluate import evaluate_split


def run_leave_category_out(X, categories, out_path, test_size=0.3, min_category_count=30,
                            only_categories=None, append=False):
    categories = np.asarray(categories)
    unique_cats = [c for c in np.unique(categories) if c != "Benign"]
    if only_categories:
        unique_cats = [c for c in unique_cats if c in only_categories]

    rows = []
    for held_out in unique_cats:
        n_held = np.sum(categories == held_out)
        if n_held < min_category_count:
            print(f"skip {held_out}: only {n_held} samples")
            continue

        is_benign = categories == "Benign"
        is_unknown = categories == held_out
        is_known_attack = (~is_benign) & (~is_unknown)

        X_benign, X_benign_test = train_test_split(
            X[is_benign], test_size=test_size, random_state=RANDOM_STATE
        )
        X_known = X[is_known_attack]
        cat_known = categories[is_known_attack]
        X_known_train, X_known_test, cat_known_train, cat_known_test = train_test_split(
            X_known, cat_known, test_size=test_size, random_state=RANDOM_STATE
        )
        X_unknown_test = X[is_unknown]

        print(f"\n=== held-out category: {held_out} "
              f"(benign_train={len(X_benign)}, known_train={len(X_known_train)}, "
              f"unknown_test={len(X_unknown_test)}) ===")

        for name, cls in ALL_DETECTORS.items():
            detector = cls()
            if name == "Prototype-Contrastive":
                fit_kwargs = {"X_known_attacks": X_known_train, "y_known": cat_known_train}
            else:
                fit_kwargs = {"X_known_attacks": X_known_train}

            X_test_all = np.vstack([X_benign_test, X_known_test, X_unknown_test])
            prof = profile_detector(detector, X_benign, X_test_all, **fit_kwargs)

            n_b, n_k, n_u = len(X_benign_test), len(X_known_test), len(X_unknown_test)
            scores = prof["scores"]
            s_benign = scores[:n_b]
            s_known = scores[n_b:n_b + n_k]
            s_unknown = scores[n_b + n_k:]

            metrics = evaluate_split(s_benign, s_known, s_unknown)

            row = {
                "held_out_category": held_out,
                "detector": name,
                "train_time_s": prof["train_time_s"],
                "infer_time_per_sample_ms": prof["infer_time_per_sample_ms"],
                "model_size_kb": prof["model_size_kb"],
                **metrics,
            }
            rows.append(row)
            print(f"  {name:22s} AUROC_unknown={metrics['auroc_unknown']:.3f}  "
                  f"UDR@5%FPR={metrics['udr_at_5pct_fpr']:.3f}  "
                  f"train={prof['train_time_s']:.2f}s  "
                  f"infer={prof['infer_time_per_sample_ms']:.4f}ms/sample  "
                  f"size={prof['model_size_kb']:.1f}KB")

    results = pd.DataFrame(rows)
    if append:
        try:
            existing = pd.read_csv(out_path)
            results = pd.concat([existing, results], ignore_index=True)
        except FileNotFoundError:
            pass
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} rows total to {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nslkdd", "ciciot2023"], required=True)
    ap.add_argument("--nslkdd-train")
    ap.add_argument("--nslkdd-test")
    ap.add_argument("--ciciot-dir")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--categories", nargs="*", default=None,
                     help="Only run these held-out categories (for chunked/incremental runs)")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if args.dataset == "nslkdd":
        df = load_nslkdd(args.nslkdd_train, args.nslkdd_test)
        X, categories, feat_names = featurize_nslkdd(df)
    else:
        df = load_ciciot2023_sampled(args.ciciot_dir, frac_per_file=args.sample_frac)
        X, categories, feat_names = featurize_ciciot2023(df)

    run_leave_category_out(X, categories, args.out, only_categories=args.categories, append=args.append)


if __name__ == "__main__":
    main()
