"""
data_prep.py
Dataset loading + preprocessing for the open-set / zero-day IDS benchmark.

Supports:
  - NSL-KDD  (used for pipeline validation; freely fetchable, has genuine
    unseen attack types in its official test split, which is convenient
    for dry-running the leave-category-out protocol before the real
    CICIoT2023 data is available)
  - CICIoT2023 (primary target dataset for the paper -- loader stub ready,
    just point CICIOT_CSV_DIR at the folder of CSVs once downloaded)

All categorical attack labels are mapped down to a small set of attack
CATEGORIES so that "leave-one-category-out" splits are meaningful (holding
out a single fine-grained label, e.g. one rare NSL-KDD attack, would leave
too few samples to be a useful zero-day simulant).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
    "difficulty",
]

# Standard NSL-KDD fine-label -> category mapping (5 categories total)
NSLKDD_CATEGORY_MAP = {
    "normal": "Benign",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS", "apache2": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (remote to local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "httptunnel": "R2L", "named": "R2L",
    "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L",
    # U2R (user to root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_nslkdd(train_path: str, test_path: str) -> pd.DataFrame:
    """Load and concatenate NSL-KDD train+test (we re-split ourselves for
    the leave-category-out protocol rather than using the official split)."""
    df_train = pd.read_csv(train_path, names=NSLKDD_COLUMNS)
    df_test = pd.read_csv(test_path, names=NSLKDD_COLUMNS)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.drop(columns=["difficulty"])
    df["category"] = df["label"].map(NSLKDD_CATEGORY_MAP)
    unmapped = df["category"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows have an unmapped label; update NSLKDD_CATEGORY_MAP")
    return df


def featurize_nslkdd(df: pd.DataFrame):
    """One-hot encode categoricals, scale numerics. Returns (X, categories, feature_names)."""
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["label", "category"]]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[cat_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    X = np.hstack([X_num, X_cat])
    feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
    categories = df["category"].values
    return X, categories, feature_names


# ---------------------------------------------------------------------
# CICIoT2023 (primary dataset -- fill in once CSVs are downloaded)
# ---------------------------------------------------------------------

# The 7 top-level attack categories used in the CICIoT2023 paper.
CICIOT_CATEGORY_MAP_HINT = {
    # fine label substrings -> category (adjust to match the exact label
    # strings present in whichever CICIoT2023 CSV variant you download --
    # the Kaggle "smaller dataset" build sometimes renames a few labels)
    "Benign": "Benign",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Recon": "Recon",
    "WebBased": "Web",
    "BruteForce": "BruteForce",
    "Spoofing": "Spoofing",
    "Mirai": "Mirai",
}


def load_ciciot2023(csv_dir: str, sample_frac: float | None = 0.05, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Load CICIoT2023 CSVs from csv_dir. Set sample_frac to down-sample for
    laptop-scale experiments (the full dataset is ~46M rows / ~13GB).

    NOTE: run once you know the exact column/label names in the file(s) you
    downloaded -- CICIoT2023 mirrors vary slightly in column naming. Inspect
    with `pd.read_csv(path, nrows=5)` first and adjust CICIOT_CATEGORY_MAP_HINT.
    """
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    frames = []
    for p in paths:
        chunk = pd.read_csv(p)
        if sample_frac is not None:
            chunk = chunk.sample(frac=sample_frac, random_state=random_state)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    return df


if __name__ == "__main__":
    import sys
    train_p, test_p = sys.argv[1], sys.argv[2]
    df = load_nslkdd(train_p, test_p)
    print(df["category"].value_counts())
    X, y, feat_names = featurize_nslkdd(df)
    print("X shape:", X.shape, "n_features:", len(feat_names))


# ---------------------------------------------------------------------
# CICIoT2023 -- full implementation (34 fine-grained labels -> 7
# top-level attack categories, following the official taxonomy from
# Neto et al. 2023). All 46 feature columns in the released CSVs are
# already numeric (binary protocol flags + statistical flow features),
# so no categorical encoding is needed -- only scaling.
# ---------------------------------------------------------------------

CICIOT_LABEL_TO_CATEGORY = {
    "BenignTraffic": "Benign",
    # DDoS (12)
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",
    # DoS (4)
    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Recon (5)
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    # Web-based (6)
    "SqlInjection": "Web", "CommandInjection": "Web", "Backdoor_Malware": "Web",
    "Uploading_Attack": "Web", "XSS": "Web", "BrowserHijacking": "Web",
    # BruteForce (1)
    "DictionaryBruteForce": "BruteForce",
    # Spoofing (2)
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Mirai (3)
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
}


def load_ciciot2023_sampled(csv_dir: str, frac_per_file: float = 0.2,
                             random_state: int = RANDOM_STATE, chunksize: int = 100_000) -> pd.DataFrame:
    """Memory-safe loader: streams each part CSV in chunks and keeps a
    random `frac_per_file` fraction of rows, so the full ~6M-row dataset
    never has to be materialized in memory at once. Maps fine-grained
    labels to the 7 top-level CICIoT2023 categories (+ Benign)."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    rng = np.random.RandomState(random_state)
    kept_frames = []
    for p in paths:
        for chunk in pd.read_csv(p, chunksize=chunksize):
            mask = rng.rand(len(chunk)) < frac_per_file
            kept_frames.append(chunk[mask])
    df = pd.concat(kept_frames, ignore_index=True)
    df["category"] = df["label"].map(CICIOT_LABEL_TO_CATEGORY)
    unmapped = df["category"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows have an unmapped CICIoT2023 label")
    return df


def featurize_ciciot2023(df: pd.DataFrame):
    """All 46 CICIoT2023 feature columns are already numeric; only
    scaling is needed. Returns (X, categories, feature_names)."""
    feature_cols = [c for c in df.columns if c not in ("label", "category")]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].astype(float))
    categories = df["category"].values
    return X, categories, feature_cols
