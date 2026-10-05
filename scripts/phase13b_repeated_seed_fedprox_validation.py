from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import scripts\\phase9_fedavg70.py. "
        "Keep this file inside the scripts folder."
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURES = 70
EXPECTED_OLD_FEATURES = 37
EXPECTED_NEW_FEATURES = 17
N_CLIENTS = 5

DEFAULT_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
DEFAULT_SCENARIO = "non_iid"

DEFAULT_ROUNDS = 10
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_MU = 0.01

LOCKED_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_locked.csv"

PHASE13A_ROOT = PROJECT_ROOT / "results" / "phase13" / "feature_search"
DEFAULT_MASK17 = PHASE13A_ROOT / "phase13_selected_mask.npy"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "phase13" / "phase13b"

OLD37_CANDIDATES = [
    PROJECT_ROOT / "results" / "phase7" / "phase7_selected_mask.npy",
    PROJECT_ROOT / "results" / "phase7" / "phase7_feature_mask.csv",
    PROJECT_ROOT / "results" / "phase7" / "phase7_selected_features.json",
    PROJECT_ROOT / "results" / "phase7" / "feature_search" / "phase7_selected_mask.npy",
    PROJECT_ROOT / "results" / "phase7" / "feature_search" / "phase7_feature_mask.csv",
    PROJECT_ROOT / "results" / "phase7" / "hybrid" / "phase7_selected_mask.npy",
    PROJECT_ROOT / "results" / "phase7" / "hybrid" / "phase7_feature_mask.csv",
    PROJECT_ROOT / "results" / "phase7" / "hybrid_evo_ga" / "phase7_selected_mask.npy",
    PROJECT_ROOT / "results" / "phase7" / "hybrid_evo_ga" / "phase7_feature_mask.csv",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase13b")


# ============================================================
# HELPERS
# ============================================================

def sep():
    print()
    print("=" * 110)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values) -> str:
    arr = np.asarray(values, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def create_run_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{stamp}"
    n = 1
    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"run_{stamp}_{n:02d}"
        n += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_class_names(label_mapping):
    inv = {int(v): str(k) for k, v in label_mapping.items()}
    return [inv[i] for i in range(len(label_mapping))]


def filter_to_project_classes(df: pd.DataFrame, label_mapping):
    canonical = {str(k).strip().lower(): str(k).strip() for k in label_mapping}
    labels = df[phase4.LABEL_COL].astype(str).str.strip()
    keep = labels.str.lower().isin(canonical)
    out = df.loc[keep].copy()
    out[phase4.LABEL_COL] = (
        out[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(canonical)
    )
    return out


def verify_feature_space(active_features, active_indices, feature_columns):
    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} active features, "
            f"found {len(active_features)}."
        )
    active_indices = np.asarray(active_indices, dtype=np.int64)
    reconstructed = [feature_columns[int(i)] for i in active_indices]
    if reconstructed != list(active_features):
        raise ValueError("Active feature/index order mismatch.")
    return active_indices


def parse_seeds(text: str):
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds are not allowed.")
    return seeds


# ============================================================
# MASK LOADING
# ============================================================

def normalize_mask(values, expected_count, source):
    arr = np.asarray(values).reshape(-1)

    if len(arr) == EXPECTED_FEATURES and arr.dtype == bool:
        mask = arr.astype(bool)

    elif len(arr) == EXPECTED_FEATURES and np.all(np.isin(arr, [0, 1])):
        mask = arr.astype(bool)

    else:
        idx = arr.astype(np.int64)
        if len(idx) != expected_count:
            raise ValueError(
                f"{source}: expected {expected_count} selected indices, "
                f"found {len(idx)}."
            )
        if idx.min() < 0 or idx.max() >= EXPECTED_FEATURES:
            raise ValueError(f"{source}: selected index outside 0..69.")
        mask = np.zeros(EXPECTED_FEATURES, dtype=bool)
        mask[idx] = True

    if int(mask.sum()) != expected_count:
        raise ValueError(
            f"{source}: expected {expected_count} selected features, "
            f"found {int(mask.sum())}."
        )
    return mask


def load_mask(path: Path, expected_count: int, active_features):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen mask not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return normalize_mask(
            np.load(path, allow_pickle=False),
            expected_count,
            str(path),
        )

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return normalize_mask(payload, expected_count, str(path))

        if isinstance(payload, dict):
            for key in (
                "mask",
                "selected_mask",
                "selected_indices",
                "selected_positions",
                "indices",
            ):
                if key in payload:
                    return normalize_mask(
                        payload[key],
                        expected_count,
                        str(path),
                    )
        raise ValueError(f"Unsupported JSON mask format: {path}")

    if suffix == ".csv":
        df = pd.read_csv(path)
        lower = {str(c).strip().lower(): c for c in df.columns}

        for key in ("selected", "is_selected", "keep", "mask"):
            if key in lower and len(df) == EXPECTED_FEATURES:
                vals = df[lower[key]].astype(str).str.strip().str.lower()
                truthy = vals.isin(["1", "true", "yes", "y"])
                falsy = vals.isin(["0", "false", "no", "n"])
                if (truthy | falsy).all():
                    return normalize_mask(
                        truthy.to_numpy(dtype=bool),
                        expected_count,
                        str(path),
                    )

        for key in ("feature_index", "index", "selected_index", "position"):
            if key in lower:
                vals = pd.to_numeric(df[lower[key]], errors="coerce").dropna()
                if len(vals) == expected_count:
                    return normalize_mask(
                        vals.to_numpy(dtype=np.int64),
                        expected_count,
                        str(path),
                    )

        for key in ("feature", "feature_name", "selected_feature", "name"):
            if key in lower:
                names = df[lower[key]].astype(str).str.strip().tolist()
                if len(names) == expected_count:
                    pos_map = {
                        str(name).strip(): i
                        for i, name in enumerate(active_features)
                    }
                    missing = [n for n in names if n not in pos_map]
                    if missing:
                        raise ValueError(
                            f"Unknown feature names in {path}: {missing[:10]}"
                        )
                    return normalize_mask(
                        [pos_map[n] for n in names],
                        expected_count,
                        str(path),
                    )

        raise ValueError(f"Could not interpret CSV mask: {path}")

    raise ValueError(f"Unsupported mask format: {path}")


def resolve_mask37(explicit):
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else PROJECT_ROOT / p

    for p in OLD37_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not auto-detect the OLD 37-feature mask.\n"
        "Run again with:\n"
        "  --mask37 <exact-path-to-old-37-feature-mask>"
    )


# ============================================================
# FAIR INITIALIZATION
# ============================================================

def build_initial_state(selected_positions, label_mapping, seed):
    selected_positions = np.asarray(selected_positions, dtype=np.int64)

    set_seed(seed + 13000)

    ref_model = phase4.IDSMLP(
        input_dim=EXPECTED_FEATURES,
        num_classes=len(label_mapping),
    )
    sel_model = phase4.IDSMLP(
        input_dim=len(selected_positions),
        num_classes=len(label_mapping),
    )

    ref_state = ref_model.state_dict()
    sel_state = sel_model.state_dict()

    candidates = [
        key
        for key, tensor in ref_state.items()
        if tensor.ndim == 2 and tensor.shape[1] == EXPECTED_FEATURES
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify first input layer: {candidates}"
        )

    first_key = candidates[0]

    for key in sel_state:
        if key == first_key:
            sliced = ref_state[key][:, selected_positions].clone()
            if sliced.shape != sel_state[key].shape:
                raise RuntimeError(
                    f"First-layer shape mismatch for {key}: "
                    f"{tuple(sliced.shape)} vs {tuple(sel_state[key].shape)}"
                )
            sel_state[key] = sliced
        else:
            if ref_state[key].shape != sel_state[key].shape:
                raise RuntimeError(f"Unexpected shape mismatch for {key}")
            sel_state[key] = ref_state[key].clone()

    del ref_model, sel_model

    return {
        k: v.detach().cpu().clone()
        for k, v in sel_state.items()
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_full_clients(
    federated_dir,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
):
    clients = []
    sep()
    print("LOADING FULL FEDERATED CLIENT DATA")

    for client_id in range(1, N_CLIENTS + 1):
        path = federated_dir / f"client_{client_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing client file: {path}")

        logger.info("Loading client %d...", client_id)
        df = pd.read_csv(path)
        df = filter_to_project_classes(df, label_mapping)

        X = phase4.transform_features(
            df=df,
            feature_columns=feature_columns,
            scaler=scaler,
            active_indices=active_indices,
        )
        y = phase4.encode_labels(df[phase4.LABEL_COL], label_mapping)

        del df
        gc.collect()

        if X.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"Client {client_id}: expected 70 features, "
                f"found {X.shape[1]}."
            )

        clients.append(
            {
                "client_id": client_id,
                "X": X.astype(np.float32, copy=False),
                "y": y.astype(np.int64, copy=False),
                "sample_count": int(len(y)),
            }
        )
        logger.info("Client %d rows=%d", client_id, len(y))

    return clients


def load_locked_test(
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
):
    if not LOCKED_TEST_FILE.exists():
        raise FileNotFoundError(f"Locked test missing: {LOCKED_TEST_FILE}")

    sep()
    print("LOADING LOCKED TEST — PHASE 13B CONFIRMATORY EVALUATION")
    print("Phase 13A feature masks are already frozen.")

    df = pd.read_csv(LOCKED_TEST_FILE)
    df = filter_to_project_classes(df, label_mapping)

    distribution = df[phase4.LABEL_COL].value_counts().to_dict()

    X = phase4.transform_features(
        df=df,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    )
    y = phase4.encode_labels(df[phase4.LABEL_COL], label_mapping)

    del df
    gc.collect()

    print(f"Locked-test rows: {len(y):,}")
    for cls, count in distribution.items():
        print(f"  {cls:25s} {int(count):,}")

    return (
        X.astype(np.float32, copy=False),
        y.astype(np.int64, copy=False),
        distribution,
    )


# ============================================================
# FEDPROX
# ============================================================

def proximal_term(model, global_parameters):
    value = torch.zeros((), device=next(model.parameters()).device)
    for local_p, global_p in zip(model.parameters(), global_parameters):
        value = value + torch.sum((local_p - global_p) ** 2)
    return value


def train_client(
    global_state,
    X,
    y,
    selected_positions,
    label_mapping,
    class_weights,
    device,
    client_id,
    round_number,
    seed,
    local_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    grad_clip,
    mu,
):
    model = phase4.IDSMLP(
        input_dim=len(selected_positions),
        num_classes=len(label_mapping),
    ).to(device)

    model.load_state_dict(global_state)

    global_parameters = [
        p.detach().clone().to(device)
        for p in model.parameters()
    ]
    for p in global_parameters:
        p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    n = len(y)

    for epoch in range(1, local_epochs + 1):
        local_seed = (
            seed
            + client_id * 100
            + round_number * 1000
            + epoch
        )

        random.seed(local_seed)
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)

        rng = np.random.default_rng(local_seed)
        order = rng.permutation(n)

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]

            xb = torch.from_numpy(
                X[idx][:, selected_positions].astype(
                    np.float32,
                    copy=False,
                )
            ).to(device)

            yb = torch.from_numpy(
                y[idx].astype(
                    np.int64,
                    copy=False,
                )
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = criterion(logits, yb)
            prox = proximal_term(model, global_parameters)
            loss = ce + (mu / 2.0) * prox
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )
            optimizer.step()

    state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    del model, global_parameters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state, n


def train_fedprox(
    clients,
    selected_positions,
    label_mapping,
    class_weights,
    device,
    seed,
    rounds,
    local_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    grad_clip,
    mu,
    method,
):
    selected_positions = np.asarray(selected_positions, dtype=np.int64)

    global_state = build_initial_state(
        selected_positions,
        label_mapping,
        seed,
    )

    start_time = time.perf_counter()

    for round_number in range(1, rounds + 1):
        print(
            f"    {method} | seed={seed} | "
            f"round={round_number}/{rounds}"
        )

        states = []
        weights = []

        for client in clients:
            state, n = train_client(
                global_state=global_state,
                X=client["X"],
                y=client["y"],
                selected_positions=selected_positions,
                label_mapping=label_mapping,
                class_weights=class_weights,
                device=device,
                client_id=int(client["client_id"]),
                round_number=round_number,
                seed=seed,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                mu=mu,
            )
            states.append(state)
            weights.append(n)

        global_state = phase4.federated_average(
            client_states=states,
            client_weights=weights,
        )

        del states, weights
        gc.collect()

    train_seconds = time.perf_counter() - start_time

    model = phase4.IDSMLP(
        input_dim=len(selected_positions),
        num_classes=len(label_mapping),
    ).to(device)
    model.load_state_dict(global_state)

    return model, float(train_seconds)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    X,
    y,
    selected_positions,
    label_mapping,
    device,
    batch_size,
):
    model.eval()
    preds = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))

        xb = torch.from_numpy(
            X[start:end][:, selected_positions].astype(
                np.float32,
                copy=False,
            )
        ).to(device)

        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_seconds = time.perf_counter() - t0
    y_pred = np.concatenate(preds)

    overall, report, cm, class_names = phase4.calculate_metrics(
        y,
        y_pred,
        label_mapping,
    )

    return overall, report, cm, class_names, float(infer_seconds)


# ============================================================
# EFFICIENCY
# ============================================================

def parameter_count(input_dim, num_classes):
    model = phase4.IDSMLP(input_dim=input_dim, num_classes=num_classes)
    n = int(sum(p.numel() for p in model.parameters()))
    del model
    return n


def efficiency_row(method, features, num_classes, rounds):
    params = parameter_count(features, num_classes)
    model_bytes = params * 4

    communication_bytes = (
        2
        * N_CLIENTS
        * rounds
        * model_bytes
    )

    return {
        "method": method,
        "features": features,
        "feature_reduction_ratio": 1.0 - features / EXPECTED_FEATURES,
        "parameter_count": params,
        "model_size_mib_float32": model_bytes / 1024 / 1024,
        "communication_bytes_raw_estimate": communication_bytes,
        "communication_mib_raw_estimate": communication_bytes / 1024 / 1024,
        "communication_formula":
            "2 * clients * rounds * parameter_count * 4 bytes",
    }


# ============================================================
# STATISTICS
# ============================================================

def summary_statistics(results):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "recall_BENIGN",
        "recall_DDoS",
        "recall_DoS GoldenEye",
        "recall_DoS Hulk",
        "recall_FTP-Patator",
        "recall_PortScan",
        "training_seconds",
        "inference_seconds",
    ]

    rows = []

    for method, group in results.groupby("method", sort=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1))
                           if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def per_class_summary(results, class_names):
    rows = []

    for method, group in results.groupby("method", sort=False):
        for class_name in class_names:
            metric = f"recall_{class_name}"
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "method": method,
                    "class": class_name,
                    "n": len(values),
                    "mean_recall": float(values.mean()),
                    "std_recall": float(values.std(ddof=1))
                                  if len(values) > 1 else 0.0,
                    "median_recall": float(values.median()),
                    "min_recall": float(values.min()),
                    "max_recall": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x - y, 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(x, y, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def paired_tests(results):
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "recall_PortScan",
        "recall_FTP-Patator",
        "recall_DoS GoldenEye",
    ]

    comparisons = [
        ("FedProx17", "FedProx70"),
        ("FedProx17", "FedProx37"),
    ]

    rows = []

    for a_name, b_name in comparisons:
        a = results[results["method"] == a_name].set_index("seed")
        b = results[results["method"] == b_name].set_index("seed")

        common = sorted(set(a.index) & set(b.index))

        for metric in metrics:
            x = a.loc[common, metric].to_numpy(dtype=np.float64)
            y = b.loc[common, metric].to_numpy(dtype=np.float64)

            statistic, p = safe_wilcoxon(x, y)
            delta = x - y

            rows.append(
                {
                    "comparison": f"{a_name}_vs_{b_name}",
                    "method_a": a_name,
                    "method_b": b_name,
                    "metric": metric,
                    "n_pairs": len(common),
                    "method_a_mean": float(np.mean(x)),
                    "method_b_mean": float(np.mean(y)),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "median_delta_a_minus_b": float(np.median(delta)),
                    "wins_a": int(np.sum(delta > 0)),
                    "ties": int(np.sum(np.isclose(delta, 0.0))),
                    "losses_a": int(np.sum(delta < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value_two_sided": p,
                    "significant_at_0_05":
                        bool(p < 0.05) if not np.isnan(p) else False,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        choices=["non_iid", "controlled_non_iid", "iid"],
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--mask37",
        default=None,
        help="Path to OLD frozen 37-feature mask.",
    )
    parser.add_argument(
        "--mask17",
        default=None,
        help="Optional Phase13A 17-feature mask override.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=DEFAULT_LOCAL_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    run_dir = create_run_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_start = time.perf_counter()

    sep()
    print("PHASE 13B — REPEATED-SEED FEDPROX VALIDATION")
    sep()
    print(f"Scenario          : {args.scenario}")
    print(f"Seeds             : {seeds}")
    print(f"Device            : {device}")
    print(f"Rounds            : {args.rounds}")
    print(f"Local epochs      : {args.local_epochs}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"Weight decay      : {args.weight_decay}")
    print(f"Gradient clip     : {args.gradient_clip}")
    print(f"FedProx mu        : {args.mu}")
    print(f"Run directory     : {run_dir}")
    print()
    print("LOCKED TEST POLICY")
    print("-" * 110)
    print("Phase 13A feature selection is finished.")
    print("17-feature mask is frozen before locked-test access.")
    print("Locked test is used only for confirmatory Phase 13B evaluation.")

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    active_indices = verify_feature_space(
        active_features,
        active_indices,
        feature_columns,
    )
    class_names = build_class_names(label_mapping)

    # --------------------------------------------------------
    # LOAD / VERIFY FROZEN MASKS BEFORE TEST ACCESS
    # --------------------------------------------------------

    mask17_path = Path(args.mask17) if args.mask17 else DEFAULT_MASK17
    if not mask17_path.is_absolute():
        mask17_path = PROJECT_ROOT / mask17_path

    mask37_path = resolve_mask37(args.mask37)

    mask17 = load_mask(
        mask17_path,
        EXPECTED_NEW_FEATURES,
        active_features,
    )
    mask37 = load_mask(
        mask37_path,
        EXPECTED_OLD_FEATURES,
        active_features,
    )
    mask70 = np.ones(EXPECTED_FEATURES, dtype=bool)

    positions70 = np.flatnonzero(mask70)
    positions37 = np.flatnonzero(mask37)
    positions17 = np.flatnonzero(mask17)

    feature_sets = {
        "FedProx70": {
            "count": len(positions70),
            "indices": positions70.tolist(),
            "features": [active_features[i] for i in positions70],
            "source": "all_70_active_features",
        },
        "FedProx37": {
            "count": len(positions37),
            "indices": positions37.tolist(),
            "features": [active_features[i] for i in positions37],
            "source": str(mask37_path),
        },
        "FedProx17": {
            "count": len(positions17),
            "indices": positions17.tolist(),
            "features": [active_features[i] for i in positions17],
            "source": str(mask17_path),
        },
    }

    save_json(
        run_dir / "phase13b_feature_sets.json",
        feature_sets,
    )

    mask_hashes = {
        "mask17_source": str(mask17_path),
        "mask17_file_sha256": sha256_file(mask17_path),
        "mask17_normalized_sha256": sha256_array(mask17),
        "mask17_selected_count": int(mask17.sum()),
        "mask37_source": str(mask37_path),
        "mask37_file_sha256": sha256_file(mask37_path),
        "mask37_normalized_sha256": sha256_array(mask37),
        "mask37_selected_count": int(mask37.sum()),
    }

    save_json(
        run_dir / "phase13b_frozen_mask_hash.json",
        mask_hashes,
    )

    shutil.copy2(
        mask17_path,
        run_dir / f"frozen17_source{mask17_path.suffix.lower()}",
    )
    shutil.copy2(
        mask37_path,
        run_dir / f"frozen37_source{mask37_path.suffix.lower()}",
    )

    sep()
    print("FROZEN FEATURE SETS VERIFIED")
    print(f"FedProx70 : {len(positions70)} features")
    print(f"FedProx37 : {len(positions37)} features")
    print(f"FedProx17 : {len(positions17)} features")
    print(f"17-mask SHA256: {mask_hashes['mask17_file_sha256']}")

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    counts = phase4.calculate_global_class_counts(
        label_mapping=label_mapping,
        chunk_size=phase4.DEFAULT_CHUNK_SIZE,
    )
    class_weights = phase4.calculate_global_class_weights(counts)

    # --------------------------------------------------------
    # FULL CLIENTS
    # --------------------------------------------------------

    federated_dir = phase4.FEDERATED_ROOT / args.scenario

    clients = load_full_clients(
        federated_dir,
        scaler,
        feature_columns,
        active_indices,
        label_mapping,
    )

    # --------------------------------------------------------
    # LOCKED TEST — FIRST ACCESSED HERE
    # --------------------------------------------------------

    X_test, y_test, test_distribution = load_locked_test(
        scaler,
        feature_columns,
        active_indices,
        label_mapping,
    )

    # --------------------------------------------------------
    # EFFICIENCY
    # --------------------------------------------------------

    efficiency = pd.DataFrame(
        [
            efficiency_row(
                "FedProx70",
                len(positions70),
                len(label_mapping),
                args.rounds,
            ),
            efficiency_row(
                "FedProx37",
                len(positions37),
                len(label_mapping),
                args.rounds,
            ),
            efficiency_row(
                "FedProx17",
                len(positions17),
                len(label_mapping),
                args.rounds,
            ),
        ]
    )

    base_params = int(
        efficiency.loc[
            efficiency["method"] == "FedProx70",
            "parameter_count",
        ].iloc[0]
    )

    efficiency["parameter_reduction_vs_70"] = (
        1.0
        -
        efficiency["parameter_count"]
        /
        base_params
    )

    efficiency.to_csv(
        run_dir / "phase13b_efficiency_comparison.csv",
        index=False,
    )

    efficiency_lookup = efficiency.set_index("method").to_dict("index")

    # --------------------------------------------------------
    # REPEATED-SEED RUNS
    # --------------------------------------------------------

    methods = [
        ("FedProx70", positions70),
        ("FedProx37", positions37),
        ("FedProx17", positions17),
    ]

    rows = []
    confusion_dir = run_dir / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=True)

    result_file = run_dir / "phase13b_all_seed_results.csv"

    for seed_no, seed in enumerate(seeds, start=1):
        sep()
        print(f"SEED {seed} ({seed_no}/{len(seeds)})")

        for method, positions in methods:
            set_seed(seed)

            print()
            print(
                f"TRAINING {method} | "
                f"features={len(positions)} | seed={seed}"
            )

            model, train_seconds = train_fedprox(
                clients=clients,
                selected_positions=positions,
                label_mapping=label_mapping,
                class_weights=class_weights,
                device=device,
                seed=seed,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip=args.gradient_clip,
                mu=args.mu,
                method=method,
            )

            (
                overall,
                report,
                cm,
                eval_class_names,
                infer_seconds,
            ) = evaluate(
                model,
                X_test,
                y_test,
                positions,
                label_mapping,
                device,
                args.batch_size,
            )

            if list(eval_class_names) != list(class_names):
                raise RuntimeError("Class order mismatch during evaluation.")

            row = {
                "seed": seed,
                "method": method,
                "features": len(positions),
                "feature_reduction_ratio":
                    1.0 - len(positions) / EXPECTED_FEATURES,
                "accuracy": float(overall["accuracy"]),
                "balanced_accuracy":
                    float(overall["balanced_accuracy"]),
                "macro_precision":
                    float(overall["macro_precision"]),
                "macro_recall":
                    float(overall["macro_recall"]),
                "macro_f1":
                    float(overall["macro_f1"]),
                "weighted_f1":
                    float(overall["weighted_f1"]),
                "training_seconds": train_seconds,
                "inference_seconds": infer_seconds,
                "parameter_count":
                    int(efficiency_lookup[method]["parameter_count"]),
                "model_size_mib_float32":
                    float(
                        efficiency_lookup[method][
                            "model_size_mib_float32"
                        ]
                    ),
                "communication_mib_raw_estimate":
                    float(
                        efficiency_lookup[method][
                            "communication_mib_raw_estimate"
                        ]
                    ),
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "fedprox_mu": args.mu,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            }

            for cls in class_names:
                row[f"recall_{cls}"] = float(report[cls]["recall"])

            rows.append(row)
            pd.DataFrame(rows).to_csv(result_file, index=False)

            pd.DataFrame(
                cm,
                index=class_names,
                columns=class_names,
            ).to_csv(
                confusion_dir
                / f"seed_{seed}_{method}_confusion_matrix.csv"
            )

            print(f"  Accuracy          : {row['accuracy']:.6f}")
            print(
                f"  Balanced Accuracy : "
                f"{row['balanced_accuracy']:.6f}"
            )
            print(f"  Macro F1          : {row['macro_f1']:.6f}")
            print(
                f"  PortScan Recall   : "
                f"{row['recall_PortScan']:.6f}"
            )
            print(
                f"  FTP Recall        : "
                f"{row['recall_FTP-Patator']:.6f}"
            )
            print(f"  Training time     : {train_seconds:.2f}s")
            print(f"  Inference time    : {infer_seconds:.2f}s")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    results = pd.DataFrame(rows)
    results.to_csv(result_file, index=False)

    stats_df = summary_statistics(results)
    stats_df.to_csv(
        run_dir / "phase13b_summary_statistics.csv",
        index=False,
    )

    class_df = per_class_summary(results, class_names)
    class_df.to_csv(
        run_dir / "phase13b_per_class_summary.csv",
        index=False,
    )

    paired_df = paired_tests(results)
    paired_df.to_csv(
        run_dir / "phase13b_paired_tests.csv",
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    sep()
    print("PHASE 13B REPEATED-SEED SUMMARY")
    sep()

    display_metrics = [
        "balanced_accuracy",
        "macro_f1",
        "recall_PortScan",
        "recall_FTP-Patator",
        "recall_DoS GoldenEye",
    ]

    display_rows = []

    for method in ("FedProx70", "FedProx37", "FedProx17"):
        group = results[results["method"] == method]
        d = {
            "method": method,
            "features": int(group["features"].iloc[0]),
        }
        for metric in display_metrics:
            d[f"{metric}_mean"] = float(group[metric].mean())
            d[f"{metric}_std"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1
                else 0.0
            )
        display_rows.append(d)

    print(pd.DataFrame(display_rows).to_string(index=False))
    print()
    print("PAIRED WILCOXON TESTS")
    print(paired_df.to_string(index=False))
    print()
    print("EFFICIENCY COMPARISON")
    print(efficiency.to_string(index=False))

    total_seconds = time.perf_counter() - total_start

    summary = {
        "phase": "13B",
        "experiment":
            "Repeated-seed FedProx confirmatory validation",
        "scenario": args.scenario,
        "seeds": seeds,
        "device": str(device),
        "locked_test": {
            "path": str(LOCKED_TEST_FILE),
            "used": True,
            "purpose":
                "confirmatory_phase13b_evaluation_only",
            "rows": len(y_test),
            "class_distribution": test_distribution,
        },
        "frozen_masks": mask_hashes,
        "feature_sets": feature_sets,
        "training_configuration": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "fedprox_mu": args.mu,
            "aggregation":
                "sample_weighted_fedavg_after_fedprox_local_training",
        },
        "efficiency":
            efficiency.to_dict(orient="records"),
        "summary_statistics":
            stats_df.to_dict(orient="records"),
        "paired_tests":
            paired_df.to_dict(orient="records"),
        "runtime_seconds": total_seconds,
        "outputs": {
            "run_directory": str(run_dir),
            "all_seed_results": str(result_file),
            "summary_statistics":
                str(run_dir / "phase13b_summary_statistics.csv"),
            "per_class_summary":
                str(run_dir / "phase13b_per_class_summary.csv"),
            "paired_tests":
                str(run_dir / "phase13b_paired_tests.csv"),
            "efficiency":
                str(run_dir / "phase13b_efficiency_comparison.csv"),
            "feature_sets":
                str(run_dir / "phase13b_feature_sets.json"),
            "mask_hash":
                str(run_dir / "phase13b_frozen_mask_hash.json"),
            "confusion_matrices": str(confusion_dir),
        },
    }

    save_json(
        run_dir / "phase13b_summary.json",
        summary,
    )

    sep()
    print("PHASE 13B COMPLETE")
    sep()
    print(f"Seeds completed         : {len(seeds)}")
    print(f"Models per seed         : 3")
    print(f"Total model runs        : {len(seeds) * 3}")
    print(f"Locked test used        : YES")
    print(f"Phase 13A mask modified : NO")
    print(f"Total runtime           : {total_seconds:.2f}s")
    print()
    print(f"Run directory:\n{run_dir}")
    print()
    print(f"All-seed results:\n{result_file}")
    print()
    print(
        "Summary JSON:\n"
        f"{run_dir / 'phase13b_summary.json'}"
    )


if __name__ == "__main__":
    main()
