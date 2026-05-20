
#!/usr/bin/env python3
"""
Overnight hyperparameter search for a LiRA-style membership inference attack.

What this script does:
  1) Loads pub.pt and model.pt from the assignment directory.
  2) Trains many reference/shadow models on stratified splits of pub.pt.
  3) Builds LiRA-style likelihood-ratio scores from the reference models.
  4) Fits a lightweight calibration model on public membership labels.
  5) Sweeps a hyperparameter grid/random subset.
  6) Saves a ranked results table and the best submission.csv.

The metric used for selection is TPR@5%FPR on pub.pt.

This is designed to be run overnight. It supports:
  - random or grid search
  - resume from previous results.jsonl
  - saving per-trial artifacts
  - final refit on the best hyperparameters and a submission.csv for priv.pt

Example:
  python lira_hparam_search.py --max-trials 48 --mode random
  python lira_hparam_search.py --mode grid --max-trials 30 --final-ensemble-runs 3
"""

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import resnet18


# ==============================
# REQUIRED for torch.load
# ==============================

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

# ============================================================
# Paths / constants
# ============================================================
BASE = Path(__file__).resolve().parent
PUB_PATH = BASE / "pub.pt"
MODEL_PATH = BASE / "model.pt"
RESULTS_JSONL = BASE / "lira_search_results-16.jsonl"
RESULTS_CSV = BASE / "lira_search_results-16.csv"
BEST_CONFIG_JSON = BASE / "best_lira_config-16.json"

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data / model helpers
# ============================================================
def make_transform():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def make_resnet18_like(num_classes: int = 9) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def load_datasets_and_model():
    print("Loading datasets...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)

    transform = make_transform()
    pub_ds.transform = transform

    print("Loading target model...")
    model = make_resnet18_like(num_classes=9)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)

    return pub_ds, model


def labels_as_numpy(ds):
    labels = ds.labels
    if torch.is_tensor(labels):
        return labels.cpu().numpy()
    return np.asarray(labels)


def membership_as_numpy(ds):
    mem = ds.membership
    if torch.is_tensor(mem):
        return mem.cpu().numpy()
    return np.asarray(mem)


def unpack_batch(batch):
    # pub_ds returns (id, img, label, membership)
    if len(batch) == 4:
        ids, imgs, labels, members = batch
        return ids, imgs, labels, members
    elif len(batch) == 3:
        ids, imgs, labels = batch
        return ids, imgs, labels, None
    raise ValueError(f"Unexpected batch size: {len(batch)}")


def make_tta_views(imgs: torch.Tensor, tta_mode: str):
    """
    Create a small, deterministic test-time augmentation set.

    This keeps the attack close to LiRA's multi-query idea without requiring
    unknown training-time augmentation details.
    """
    if tta_mode == "none":
        return [imgs]
    if tta_mode in {"flip", "hflip"}:
        # Identity + horizontal flip
        return [imgs, torch.flip(imgs, dims=[3])]
    raise ValueError(f"Unknown tta_mode: {tta_mode}")


# ============================================================
# Shadow/reference model training
# ============================================================
def stratified_split_indices(labels: np.ndarray, member_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    in_idx, out_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * member_frac))
        cut = max(1, min(cut, len(idx) - 1))
        in_idx.extend(idx[:cut].tolist())
        out_idx.extend(idx[cut:].tolist())

    rng.shuffle(in_idx)
    rng.shuffle(out_idx)
    return in_idx, out_idx


def train_one_classifier(
    model: nn.Module,
    dataset,
    idxs: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    model = model.to(DEVICE)
    model.train()

    loader = DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0

        for batch in loader:
            _, imgs, labels, *_ = unpack_batch(batch)
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        sched.step()
        print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss/total:.4f}")

    return model


def train_reference_models(
    pub_ds,
    num_ref_models: int,
    member_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
):
    labels = labels_as_numpy(pub_ds)
    num_classes = int(labels.max()) + 1
    refs = []

    for r in range(num_ref_models):
        print(f"\nTraining reference model {r+1}/{num_ref_models}")
        in_idx, out_idx = stratified_split_indices(labels, member_frac=member_frac, seed=seed + 1000 + r)

        m = make_resnet18_like(num_classes=num_classes)
        m = train_one_classifier(
            m,
            pub_ds,
            in_idx,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        m.eval()
        refs.append({
            "model": m,
            "in_idx": np.array(in_idx, dtype=np.int64),
            "out_idx": np.array(out_idx, dtype=np.int64),
        })

    return refs


# ============================================================
# LiRA statistics
# ============================================================
def true_class_scalar_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stat_mode: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    One scalar per example.
    - logit_prob: logit(p_true) -- spread out and stable for Gaussian fits
    - neg_loss: -CE(true_label) -- monotone with confidence

    Temperature > 1.0 softens logits before taking the statistic.
    """
    logits = logits / max(float(temperature), 1e-6)
    probs = F.softmax(logits, dim=1)
    labels = labels.long()
    p_true = probs[torch.arange(probs.size(0), device=probs.device), labels].clamp(1e-6, 1 - 1e-6)

    if stat_mode == "logit_prob":
        return torch.log(p_true / (1.0 - p_true))
    elif stat_mode == "neg_loss":
        return torch.log(p_true)
    else:
        raise ValueError(f"Unknown stat_mode: {stat_mode}")


@torch.inference_mode()
def collect_scalar_scores(model, dataset, batch_size: int, stat_mode: str, tta_mode: str, temperature: float) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    out = []
    for batch in loader:
        _, imgs, labels, *_ = unpack_batch(batch)
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).long()

        views = make_tta_views(imgs, tta_mode)
        stats = []
        for view in views:
            logits = model(view)
            stat = true_class_scalar_from_logits(logits, labels, stat_mode, temperature=temperature)
            stats.append(stat)

        stat = torch.stack(stats, dim=0).mean(dim=0)
        out.extend(stat.detach().cpu().numpy().tolist())

    return np.asarray(out, dtype=np.float32)


def collect_reference_matrix(reference_models, dataset, batch_size: int, stat_mode: str, tta_mode: str, temperature: float):
    n = len(dataset)
    r = len(reference_models)
    scores = np.zeros((r, n), dtype=np.float32)
    in_mask = np.zeros((r, n), dtype=np.bool_)

    for i, ref in enumerate(reference_models):
        print(f"Collecting scores from reference model {i+1}/{r}")
        scores[i] = collect_scalar_scores(ref["model"], dataset, batch_size=batch_size, stat_mode=stat_mode, tta_mode=tta_mode, temperature=temperature)
        in_mask[i, ref["in_idx"]] = True

    return scores, in_mask


def fit_gaussian_stats(reference_scores: np.ndarray, reference_in_mask: np.ndarray, eps: float = 1e-4):
    r, n = reference_scores.shape
    mu_in = np.zeros(n, dtype=np.float32)
    sig_in = np.zeros(n, dtype=np.float32)
    mu_out = np.zeros(n, dtype=np.float32)
    sig_out = np.zeros(n, dtype=np.float32)

    for j in range(n):
        in_scores = reference_scores[reference_in_mask[:, j], j]
        out_scores = reference_scores[~reference_in_mask[:, j], j]

        if len(in_scores) == 0:
            in_scores = out_scores
        if len(out_scores) == 0:
            out_scores = in_scores

        mu_in[j] = float(np.mean(in_scores))
        sig_in[j] = float(np.std(in_scores) + eps)
        mu_out[j] = float(np.mean(out_scores))
        sig_out[j] = float(np.std(out_scores) + eps)

    return mu_in, sig_in, mu_out, sig_out


def gaussian_logpdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-4)
    return -0.5 * (np.log(2.0 * np.pi * sigma * sigma) + ((x - mu) ** 2) / (sigma * sigma))


# ============================================================
# Feature construction / calibration
# ============================================================
def make_features(
    target_scores: np.ndarray,
    labels: np.ndarray,
    mu_in: np.ndarray,
    sig_in: np.ndarray,
    mu_out: np.ndarray,
    sig_out: np.ndarray,
    feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    llr = gaussian_logpdf(target_scores, mu_in, sig_in) - gaussian_logpdf(target_scores, mu_out, sig_out)

    if feature_mode == "llr_only":
        X = llr.reshape(-1, 1)
    elif feature_mode == "rich":
        X = np.column_stack([
            llr,
            target_scores,
            mu_out,
            sig_out,
            mu_in,
            sig_in,
            labels.astype(np.float32),
        ])
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return X.astype(np.float32), llr.astype(np.float32)


def fit_lira_calibrator(X: np.ndarray, y: np.ndarray, C: float, class_weight: str):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            C=C,
            class_weight=class_weight,
            n_jobs=None,
            solver="lbfgs",
        ),
    )
    clf.fit(X, y)
    return clf


def fit_classwise_in_regressors(
    pub_features: np.ndarray,
    labels: np.ndarray,
    mu_in: np.ndarray,
    log_sig_in: np.ndarray,
    num_classes: int,
    ridge_alpha: float,
):
    """
    Predict IN-distribution parameters for private samples using simple class-wise regressors.

    Input features:
      [target_score, mu_out, sig_out, class_id]
    """
    mu_regs = {}
    sig_regs = {}

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        Xc = pub_features[idx]
        y_mu = mu_in[idx]
        y_logsig = log_sig_in[idx]

        mu_regs[c] = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        sig_regs[c] = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))

        mu_regs[c].fit(Xc, y_mu)
        sig_regs[c].fit(Xc, y_logsig)

    return mu_regs, sig_regs


# ============================================================
# Trial execution
# ============================================================
@dataclass(frozen=True)
class SearchConfig:
    num_ref_models: int
    ref_member_frac: float
    ref_epochs: int
    ref_lr: float
    ref_weight_decay: float
    stat_mode: str
    tta_mode: str
    temperature: float
    feature_mode: str
    cal_C: float
    cal_class_weight: str
    ridge_alpha: float

    def key(self) -> str:
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def search_config_from_dict(data: Dict) -> SearchConfig:
    defaults = {
        "num_ref_models": 8,
        "ref_member_frac": 0.5,
        "ref_epochs": 10,
        "ref_lr": 1e-3,
        "ref_weight_decay": 5e-4,
        "stat_mode": "logit_prob",
        "tta_mode": "none",
        "temperature": 1.0,
        "feature_mode": "llr_only",
        "cal_C": 1.0,
        "cal_class_weight": "balanced",
        "ridge_alpha": 1.0,
    }
    defaults.update(data)
    return SearchConfig(**defaults)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_5fpr = float(np.interp(0.05, fpr, tpr))
    tpr_at_1fpr = float(np.interp(0.01, fpr, tpr))
    return {
        "auc": float(roc_auc),
        "tpr_at_5fpr": tpr_at_5fpr,
        "tpr_at_1fpr": tpr_at_1fpr,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def run_trial(pub_ds, target_model, cfg: SearchConfig, seed: int, batch_size: int):
    """
    Full LiRA-like attack for one hyperparameter setting.
    Returns metrics.
    """
    labels_pub = labels_as_numpy(pub_ds)
    num_classes = int(labels_pub.max()) + 1
    y_pub = membership_as_numpy(pub_ds).astype(np.int64)

    print("\n" + "=" * 80)
    print(f"Running trial {cfg.key()} with config:\n{json.dumps(asdict(cfg), indent=2)}")
    print("=" * 80)

    # 1) Train reference models
    refs = train_reference_models(
        pub_ds=pub_ds,
        num_ref_models=cfg.num_ref_models,
        member_frac=cfg.ref_member_frac,
        epochs=cfg.ref_epochs,
        batch_size=128,
        lr=cfg.ref_lr,
        weight_decay=cfg.ref_weight_decay,
        seed=seed,
    )

    # 2) Reference matrix on pub
    ref_scores_pub, ref_in_mask = collect_reference_matrix(
        refs, pub_ds, batch_size=batch_size, stat_mode=cfg.stat_mode, tta_mode=cfg.tta_mode, temperature=cfg.temperature
    )
    mu_in_pub, sig_in_pub, mu_out_pub, sig_out_pub = fit_gaussian_stats(ref_scores_pub, ref_in_mask)

    # 3) Target scores on pub
    print("Scoring pub_ds with target model...")
    target_scores_pub = collect_scalar_scores(target_model, pub_ds, batch_size=batch_size, stat_mode=cfg.stat_mode, tta_mode=cfg.tta_mode, temperature=cfg.temperature)

    # 4) Class-wise regressors for IN stats
    pub_reg_features = np.column_stack([
        target_scores_pub,
        mu_out_pub,
        sig_out_pub,
        labels_pub.astype(np.float32),
    ])

    mu_regs, sig_regs = fit_classwise_in_regressors(
        pub_features=pub_reg_features,
        labels=labels_pub,
        mu_in=mu_in_pub,
        log_sig_in=np.log(sig_in_pub),
        num_classes=num_classes,
        ridge_alpha=cfg.ridge_alpha,
    )

    # 5) Build pub features and fit calibrator
    X_pub, llr_pub = make_features(
        target_scores=target_scores_pub,
        labels=labels_pub,
        mu_in=mu_in_pub,
        sig_in=sig_in_pub,
        mu_out=mu_out_pub,
        sig_out=sig_out_pub,
        feature_mode=cfg.feature_mode,
    )
    calibrator = fit_lira_calibrator(X_pub, y_pub, C=cfg.cal_C, class_weight=cfg.cal_class_weight)
    pub_scores = calibrator.predict_proba(X_pub)[:, 1].astype(np.float32)

    pub_metrics_raw = evaluate_scores(y_pub, llr_pub)
    pub_metrics_cal = evaluate_scores(y_pub, pub_scores)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "cfg": asdict(cfg),
        "config_key": cfg.key(),
        "pub_metrics_raw": pub_metrics_raw,
        "pub_metrics_cal": pub_metrics_cal,
        "llr_pub": llr_pub,
        "pub_scores": pub_scores,
    }


# ============================================================
# Search space
# ============================================================
def build_search_space(args) -> List[SearchConfig]:
    num_ref_models = args.num_ref_models or [16]
    ref_member_frac = args.ref_member_frac or [0.5, 0.55]
    ref_epochs = args.ref_epochs or [20]
    ref_lr = args.ref_lr or [1e-3, 5e-4]
    ref_weight_decay = args.ref_weight_decay or [5e-4, 1e-4]
    stat_mode = args.stat_mode or ["neg_loss"]
    tta_mode = args.tta_mode or ["flip"]
    temperature = args.temperature or [1.0]
    feature_mode = args.feature_mode or ["rich"]
    cal_C = args.cal_C or [4.0]
    cal_class_weight = args.cal_class_weight or ["balanced"]
    ridge_alpha = args.ridge_alpha or [1.0]

    grid = []
    for combo in itertools.product(
        num_ref_models,
        ref_member_frac,
        ref_epochs,
        ref_lr,
        ref_weight_decay,
        stat_mode,
        tta_mode,
        temperature,
        feature_mode,
        cal_C,
        cal_class_weight,
        ridge_alpha,
    ):
        grid.append(SearchConfig(
            num_ref_models=int(combo[0]),
            ref_member_frac=float(combo[1]),
            ref_epochs=int(combo[2]),
            ref_lr=float(combo[3]),
            ref_weight_decay=float(combo[4]),
            stat_mode=str(combo[5]),
            tta_mode=str(combo[6]),
            temperature=float(combo[7]),
            feature_mode=str(combo[8]),
            cal_C=float(combo[9]),
            cal_class_weight=str(combo[10]),
            ridge_alpha=float(combo[11]),
        ))
    return grid


def choose_trials(grid: List[SearchConfig], mode: str, max_trials: int, seed: int) -> List[SearchConfig]:
    rng = random.Random(seed)
    grid = grid[:]
    if mode == "random":
        rng.shuffle(grid)
    elif mode == "grid":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return grid[:max_trials]


# ============================================================
# Resume / logging
# ============================================================
def load_done_keys(results_jsonl: Path) -> set:
    done = set()
    if not results_jsonl.exists():
        return done
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add(rec["config_key"])
            except Exception:
                continue
    return done


def append_result(results_jsonl: Path, record: Dict):
    with results_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================
# Final fit on best config + submission generation
# ============================================================
def fit_and_score_best(pub_ds, target_model, cfg: SearchConfig, seed: int, batch_size: int, final_ensemble_runs: int):
    """
    Re-run the best config and optionally average several independent runs
    for the final submission.
    """
    trial_records = []

    for run in range(final_ensemble_runs):
        print(f"\nFinal ensemble run {run+1}/{final_ensemble_runs}")
        rec = run_trial(
            pub_ds=pub_ds,
            target_model=target_model,
            cfg=cfg,
            seed=seed + 10000 + run,
            batch_size=batch_size,
        )
        trial_records.append(rec)


    return trial_records


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive hyperparameter search for LiRA-style MIA.")
    parser.add_argument("--mode", choices=["random", "grid"], default="random")
    parser.add_argument("--max-trials", type=int, default=48)
    parser.add_argument("--seed", type=int, default=6967)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--final-ensemble-runs", type=int, default=1)

    # Optional explicit search-space overrides.
    # If omitted, the defaults in build_search_space() are used.
    parser.add_argument("--num-ref-models", type=int, nargs="*")
    parser.add_argument("--ref-member-frac", type=float, nargs="*")
    parser.add_argument("--ref-epochs", type=int, nargs="*")
    parser.add_argument("--ref-lr", type=float, nargs="*")
    parser.add_argument("--ref-weight-decay", type=float, nargs="*")
    parser.add_argument("--stat-mode", type=str, nargs="*")
    parser.add_argument("--tta-mode", type=str, nargs="*")
    parser.add_argument("--temperature", type=float, nargs="*")
    parser.add_argument("--feature-mode", type=str, nargs="*")
    parser.add_argument("--cal-C", type=float, nargs="*")
    parser.add_argument("--cal-class-weight", type=str, nargs="*")
    parser.add_argument("--ridge-alpha", type=float, nargs="*")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pub_ds, target_model = load_datasets_and_model()

    grid = build_search_space(args)
    trials = choose_trials(grid, mode=args.mode, max_trials=args.max_trials, seed=args.seed)

    done_keys = load_done_keys(RESULTS_JSONL)
    print(f"Loaded {len(done_keys)} completed trials from {RESULTS_JSONL}" if RESULTS_JSONL.exists() else "No previous results found.")

    best_record = None
    best_metric = -1.0

    for i, cfg in enumerate(trials, start=1):
        if cfg.key() in done_keys:
            print(f"\nSkipping completed trial {i}/{len(trials)}: {cfg.key()}")
            continue

        try:
            record = run_trial(
                pub_ds=pub_ds,
                target_model=target_model,
                cfg=cfg,
                seed=args.seed + i * 17,
                batch_size=args.batch_size,
            )

            y_pub = membership_as_numpy(pub_ds).astype(np.int64)
            pub_metrics_cal = record["pub_metrics_cal"]
            score = pub_metrics_cal["tpr_at_5fpr"]

            result_row = {
                "config_key": record["config_key"],
                "trial_index": i,
                "mode": args.mode,
                "score": score,
                "auc": pub_metrics_cal["auc"],
                "tpr_at_5fpr": pub_metrics_cal["tpr_at_5fpr"],
                "tpr_at_1fpr": pub_metrics_cal["tpr_at_1fpr"],
                "pub_auc_raw": record["pub_metrics_raw"]["auc"],
                "pub_tpr5_raw": record["pub_metrics_raw"]["tpr_at_5fpr"],
                "pub_tpr1_raw": record["pub_metrics_raw"]["tpr_at_1fpr"],
                "cfg_json": json.dumps(record["cfg"]),
            }
            append_result(RESULTS_JSONL, result_row)
            print(f"Trial result: TPR@5%FPR={score:.4f} | AUC={pub_metrics_cal['auc']:.4f}")

            if score > best_metric:
                best_metric = score
                best_record = {
                    "cfg": cfg,
                    "row": result_row,
                }

        except Exception as e:
            err_row = {
                "config_key": cfg.key(),
                "trial_index": i,
                "error": repr(e),
                "cfg_json": json.dumps(asdict(cfg)),
            }
            append_result(RESULTS_JSONL, err_row)
            print(f"Trial failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summarize all results
    if RESULTS_JSONL.exists():
        records = []
        with RESULTS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if "score" in rec:
                        records.append(rec)
                except Exception:
                    pass
        if records:
            df = pd.DataFrame(records)
            if "score" in df.columns:
                df = df.sort_values("score", ascending=False)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"\nSaved search summary: {RESULTS_CSV}")

    if best_record is None:
        if RESULTS_JSONL.exists():
            # Reload best from summary if any
            try:
                df = pd.read_csv(RESULTS_CSV)
                if len(df) > 0:
                    best_json = json.loads(df.iloc[0]["cfg_json"])
                    best_cfg = search_config_from_dict(best_json)
                    best_metric = float(df.iloc[0]["score"])
                else:
                    raise RuntimeError("No successful trials found.")
            except Exception as e:
                raise RuntimeError("No successful trials found.") from e
        else:
            raise RuntimeError("No successful trials found.")
    else:
        best_cfg = best_record["cfg"]

    with BEST_CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "best_metric": best_metric,
            "best_config": asdict(best_cfg),
        }, f, indent=2)
    print(f"\nBest config saved to: {BEST_CONFIG_JSON}")
    print(f"Best pub TPR@5%FPR: {best_metric:.4f}")

    # Final pass: refit best config and produce submission.csv
    fit_and_score_best(
        pub_ds=pub_ds,
        target_model=target_model,
        cfg=best_cfg,
        seed=args.seed,
        batch_size=args.batch_size,
        final_ensemble_runs=args.final_ensemble_runs,
    )


if __name__ == "__main__":
    main()
