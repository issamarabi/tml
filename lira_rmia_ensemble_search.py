#!/usr/bin/env python3
"""
Search a two-model ensemble over fixed LiRA and RMIA attacks.

This script:
  1) Loads pub.pt, priv.pt, and model.pt.
  2) Runs a fixed LiRA attack for one or more seeds.
  3) Runs a fixed RMIA attack for one or more seeds.
  4) Searches ensemble weights over the two attack outputs.
  5) Selects the best weight setting on pub.pt by TPR@5%FPR.
  6) Writes submission.csv using the best weight setting on priv.pt.

The ensemble search is cheap relative to the attacks themselves:
we cache LiRA/RMIA score vectors per seed once, then sweep weights on top.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate
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
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission-4.csv"
RESULTS_JSONL = BASE / "lira_rmia_ensemble_results-4.jsonl"
RESULTS_CSV = BASE / "lira_rmia_ensemble_results-4.csv"
BEST_CONFIG_JSON = BASE / "best_lira_rmia_ensemble-4.json"

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
RMIA_AUG_SETS = {
    "1q": ("identity",),
    "2q": ("identity", "hflip"),
    "5q": ("identity", "shift_left", "shift_right", "shift_up", "shift_down"),
    "6q": ("identity", "hflip", "shift_left", "shift_right", "shift_up", "shift_down"),
}


@dataclass(frozen=True)
class FixedLiRAConfig:
    num_ref_models: int = 16
    ref_member_frac: float = 0.55
    ref_epochs: int = 20
    ref_lr: float = 1e-3
    ref_weight_decay: float = 5e-4
    stat_mode: str = "neg_loss"
    feature_mode: str = "rich"
    cal_C: float = 2.0
    cal_class_weight: str = "balanced"
    ridge_alpha: float = 1.0


@dataclass(frozen=True)
class FixedRMIAConfig:
    num_ref_models: int = 16
    ref_member_frac: float = 0.5
    ref_epoch: int = 15
    pool_size: int = 500
    gamma: float = 0.5
    out_scale: float = 1.0
    temperature: float = 1.75
    aug_mode: str = "6q"
    agg_mode: str = "median"
    ref_lr: float = 1e-3
    ref_weight_decay: float = 5e-4


@dataclass(frozen=True)
class EnsembleWeightConfig:
    lira_weight: float
    rmia_weight: float

    def key(self) -> str:
        return f"lira={self.lira_weight:.6f}|rmia={self.rmia_weight:.6f}"


LIRA_CFG = FixedLiRAConfig()
RMIA_CFG = FixedRMIAConfig()


# ============================================================
# Basic utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def labels_as_numpy(ds) -> np.ndarray:
    labels = ds.labels
    if torch.is_tensor(labels):
        return labels.cpu().numpy()
    return np.asarray(labels)


def membership_as_numpy(ds) -> np.ndarray:
    members = ds.membership
    if torch.is_tensor(members):
        return members.cpu().numpy()
    return np.asarray(members)


def unpack_batch(batch):
    if len(batch) == 4:
        ids, imgs, labels, members = batch
        return ids, imgs, labels, members
    if len(batch) == 3:
        ids, imgs, labels = batch
        return ids, imgs, labels, None
    raise ValueError(f"Unexpected batch size: {len(batch)}")


def collate_with_optional_nones(batch):
    """
    Like PyTorch's default collate, but it preserves a trailing None field.
    """
    first = batch[0]
    if len(first) == 4:
        ids, imgs, labels, members = zip(*batch)
        return (
            list(ids),
            default_collate(imgs),
            None if all(label is None for label in labels) else default_collate(labels),
            None if all(member is None for member in members) else default_collate(members),
        )
    if len(first) == 3:
        ids, imgs, labels = zip(*batch)
        return (
            list(ids),
            default_collate(imgs),
            None if all(label is None for label in labels) else default_collate(labels),
        )
    raise ValueError(f"Unexpected sample size: {len(first)}")


def labels_to_numpy_or_none(ds):
    if not hasattr(ds, "labels"):
        return None
    labels = ds.labels
    if labels is None:
        return None
    try:
        arr = np.asarray(labels)
        if arr.dtype == object and any(label is None for label in arr[: min(len(arr), 8)]):
            return None
        return arr.astype(np.int64, copy=False)
    except Exception:
        return None


@torch.inference_mode()
def collect_pred_labels(model, dataset, batch_size: int, num_workers: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_optional_nones,
    )

    preds = []
    for batch in loader:
        _, imgs, _, *_ = unpack_batch(batch)
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(preds, axis=0)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_5 = float(np.interp(0.05, fpr, tpr))
    tpr_at_1 = float(np.interp(0.01, fpr, tpr))
    return {
        "auc": float(roc_auc),
        "tpr_at_5fpr": tpr_at_5,
        "tpr_at_1fpr": tpr_at_1,
    }


def make_eval_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def make_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def make_model(num_classes: int = 9) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def load_datasets_and_model():
    print("Loading datasets...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    priv_ds = torch.load(PRIV_PATH, weights_only=False)

    eval_transform = make_eval_transform()
    pub_ds.transform = eval_transform
    priv_ds.transform = eval_transform

    num_classes = int(labels_as_numpy(pub_ds).max()) + 1

    print("Loading target model...")
    model = make_model(num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)

    return pub_ds, priv_ds, model


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


# ============================================================
# LiRA
# ============================================================
def train_one_classifier(
    model: nn.Module,
    dataset,
    idxs: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
) -> nn.Module:
    model = model.to(DEVICE)
    model.train()

    loader = DataLoader(
        Subset(dataset, idxs),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
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

            total_loss += float(loss.item()) * imgs.size(0)
            total += imgs.size(0)

        sched.step()
        print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss / max(total, 1):.4f}")

    return model


def train_lira_reference_models(
    pub_ds,
    cfg: FixedLiRAConfig,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    labels = labels_as_numpy(pub_ds)
    num_classes = int(labels.max()) + 1
    refs = []
    train_transform = make_train_transform()
    eval_transform = make_eval_transform()

    try:
        for r in range(cfg.num_ref_models):
            print(f"\n[LiRA] Training reference model {r+1}/{cfg.num_ref_models}")
            in_idx, out_idx = stratified_split_indices(
                labels,
                member_frac=cfg.ref_member_frac,
                seed=seed + 1000 + r,
            )

            pub_ds.transform = train_transform
            model = make_model(num_classes=num_classes)
            model = train_one_classifier(
                model,
                pub_ds,
                in_idx,
                epochs=cfg.ref_epochs,
                batch_size=batch_size,
                lr=cfg.ref_lr,
                weight_decay=cfg.ref_weight_decay,
                num_workers=num_workers,
            )
            model.eval()
            refs.append({
                "model": model,
                "in_idx": np.asarray(in_idx, dtype=np.int64),
                "out_idx": np.asarray(out_idx, dtype=np.int64),
            })
    finally:
        pub_ds.transform = eval_transform

    return refs


def true_class_scalar_from_logits(logits: torch.Tensor, labels: torch.Tensor | None, stat_mode: str) -> torch.Tensor:
    if labels is None:
        labels = torch.argmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    labels = labels.long()
    p_true = probs[torch.arange(probs.size(0), device=probs.device), labels].clamp(1e-6, 1.0 - 1e-6)

    if stat_mode == "logit_prob":
        return torch.log(p_true / (1.0 - p_true))
    if stat_mode == "neg_loss":
        return torch.log(p_true)
    raise ValueError(f"Unknown stat_mode: {stat_mode}")


@torch.inference_mode()
def collect_scalar_scores(model, dataset, batch_size: int, stat_mode: str, num_workers: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_optional_nones,
    )

    out = []
    for batch in loader:
        _, imgs, labels, *_ = unpack_batch(batch)
        imgs = imgs.to(DEVICE)
        labels = None if labels is None else torch.as_tensor(labels, device=DEVICE).long()
        logits = model(imgs)
        stat = true_class_scalar_from_logits(logits, labels, stat_mode)
        out.append(stat.detach().cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(out, axis=0)


def collect_lira_reference_matrix(reference_models, dataset, batch_size: int, stat_mode: str, num_workers: int):
    n = len(dataset)
    r = len(reference_models)
    scores = np.zeros((r, n), dtype=np.float32)
    in_mask = np.zeros((r, n), dtype=np.bool_)

    for i, ref in enumerate(reference_models):
        print(f"[LiRA] Collecting scalar scores from reference model {i+1}/{r}")
        scores[i] = collect_scalar_scores(
            ref["model"],
            dataset,
            batch_size=batch_size,
            stat_mode=stat_mode,
            num_workers=num_workers,
        )
        if "in_idx" in ref:
            in_mask[i, ref["in_idx"]] = True

    return scores, in_mask


def fit_gaussian_stats(reference_scores: np.ndarray, reference_in_mask: np.ndarray, eps: float = 1e-4):
    _, n = reference_scores.shape
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


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-4)
    return -0.5 * (np.log(2.0 * np.pi * sigma * sigma) + ((x - mu) ** 2) / (sigma * sigma))


def make_lira_features(
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
        x = llr.reshape(-1, 1)
    elif feature_mode == "rich":
        x = np.column_stack([
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

    return x.astype(np.float32), llr.astype(np.float32)


def fit_lira_calibrator(x: np.ndarray, y: np.ndarray, C: float, class_weight: str):
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
    clf.fit(x, y)
    return clf


def fit_classwise_in_regressors(
    pub_features: np.ndarray,
    labels: np.ndarray,
    mu_in: np.ndarray,
    log_sig_in: np.ndarray,
    num_classes: int,
    ridge_alpha: float,
):
    mu_regs = {}
    sig_regs = {}

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        x_c = pub_features[idx]
        y_mu = mu_in[idx]
        y_logsig = log_sig_in[idx]

        mu_regs[c] = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        sig_regs[c] = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        mu_regs[c].fit(x_c, y_mu)
        sig_regs[c].fit(x_c, y_logsig)

    return mu_regs, sig_regs


def predict_private_in_stats(
    priv_features: np.ndarray,
    priv_labels: np.ndarray,
    mu_regs: Dict[int, object],
    sig_regs: Dict[int, object],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_in_hat = np.zeros(len(priv_labels), dtype=np.float32)
    sig_in_hat = np.zeros(len(priv_labels), dtype=np.float32)

    for c in range(num_classes):
        idx = np.where(priv_labels == c)[0]
        if len(idx) == 0:
            continue
        x_c = priv_features[idx]
        mu_in_hat[idx] = mu_regs[c].predict(x_c).astype(np.float32)
        sig_in_hat[idx] = np.exp(sig_regs[c].predict(x_c)).astype(np.float32)

    sig_in_hat = np.maximum(sig_in_hat, 1e-4)
    return mu_in_hat, sig_in_hat


def run_lira_attack(
    pub_ds,
    priv_ds,
    target_model,
    cfg: FixedLiRAConfig,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    print("\n" + "=" * 80)
    print(f"[LiRA] Seed {seed}")
    print("=" * 80)

    labels_pub = labels_as_numpy(pub_ds)
    labels_priv = labels_to_numpy_or_none(priv_ds)
    members_pub = membership_as_numpy(pub_ds).astype(np.int64)
    num_classes = int(labels_pub.max()) + 1

    refs = train_lira_reference_models(
        pub_ds=pub_ds,
        cfg=cfg,
        seed=seed,
        batch_size=128,
        num_workers=num_workers,
    )

    eval_transform = make_eval_transform()
    pub_ds.transform = eval_transform
    priv_ds.transform = eval_transform

    ref_scores_pub, ref_in_mask = collect_lira_reference_matrix(
        refs,
        pub_ds,
        batch_size=batch_size,
        stat_mode=cfg.stat_mode,
        num_workers=num_workers,
    )
    mu_in_pub, sig_in_pub, mu_out_pub, sig_out_pub = fit_gaussian_stats(ref_scores_pub, ref_in_mask)

    print("[LiRA] Scoring target model on pub.pt")
    target_scores_pub = collect_scalar_scores(
        target_model,
        pub_ds,
        batch_size=batch_size,
        stat_mode=cfg.stat_mode,
        num_workers=num_workers,
    )

    if labels_priv is None:
        labels_priv = collect_pred_labels(
            target_model,
            priv_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    pub_reg_features = np.column_stack([
        target_scores_pub,
        mu_out_pub,
        sig_out_pub,
        labels_pub.astype(np.float32),
    ]).astype(np.float32)
    mu_regs, sig_regs = fit_classwise_in_regressors(
        pub_features=pub_reg_features,
        labels=labels_pub,
        mu_in=mu_in_pub,
        log_sig_in=np.log(sig_in_pub),
        num_classes=num_classes,
        ridge_alpha=cfg.ridge_alpha,
    )

    x_pub, llr_pub = make_lira_features(
        target_scores=target_scores_pub,
        labels=labels_pub,
        mu_in=mu_in_pub,
        sig_in=sig_in_pub,
        mu_out=mu_out_pub,
        sig_out=sig_out_pub,
        feature_mode=cfg.feature_mode,
    )
    calibrator = fit_lira_calibrator(
        x_pub,
        members_pub,
        C=cfg.cal_C,
        class_weight=cfg.cal_class_weight,
    )
    pub_scores = calibrator.predict_proba(x_pub)[:, 1].astype(np.float32)

    print("[LiRA] Collecting OUT statistics for priv.pt")
    ref_scores_priv = np.zeros((len(refs), len(priv_ds)), dtype=np.float32)
    for i, ref in enumerate(refs):
        ref_scores_priv[i] = collect_scalar_scores(
            ref["model"],
            priv_ds,
            batch_size=batch_size,
            stat_mode=cfg.stat_mode,
            num_workers=num_workers,
        )
    mu_out_priv = ref_scores_priv.mean(axis=0).astype(np.float32)
    sig_out_priv = (ref_scores_priv.std(axis=0) + 1e-4).astype(np.float32)

    print("[LiRA] Scoring target model on priv.pt")
    target_scores_priv = collect_scalar_scores(
        target_model,
        priv_ds,
        batch_size=batch_size,
        stat_mode=cfg.stat_mode,
        num_workers=num_workers,
    )
    priv_features_for_reg = np.column_stack([
        target_scores_priv,
        mu_out_priv,
        sig_out_priv,
        labels_priv.astype(np.float32),
    ]).astype(np.float32)
    mu_in_priv_hat, sig_in_priv_hat = predict_private_in_stats(
        priv_features=priv_features_for_reg,
        priv_labels=labels_priv,
        mu_regs=mu_regs,
        sig_regs=sig_regs,
        num_classes=num_classes,
    )

    x_priv, _ = make_lira_features(
        target_scores=target_scores_priv,
        labels=labels_priv,
        mu_in=mu_in_priv_hat,
        sig_in=sig_in_priv_hat,
        mu_out=mu_out_priv,
        sig_out=sig_out_priv,
        feature_mode=cfg.feature_mode,
    )
    priv_scores = calibrator.predict_proba(x_priv)[:, 1].astype(np.float32)

    metrics_pub = evaluate_scores(members_pub, pub_scores)
    metrics_llr = evaluate_scores(members_pub, llr_pub)

    for ref in refs:
        del ref["model"]
    del refs, ref_scores_pub, ref_scores_priv, ref_in_mask, calibrator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "[LiRA] "
        f"pub TPR@5%FPR={metrics_pub['tpr_at_5fpr']:.4f} | "
        f"AUC={metrics_pub['auc']:.4f}"
    )

    return {
        "pub_scores": pub_scores,
        "priv_scores": priv_scores,
        "pub_metrics_cal": metrics_pub,
        "pub_metrics_raw": metrics_llr,
    }


# ============================================================
# RMIA
# ============================================================
def shift_tensor_batch(imgs: torch.Tensor, dx: int, dy: int, pad: int = 2) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return imgs

    _, _, h, w = imgs.shape
    padded = F.pad(imgs, (pad, pad, pad, pad), mode="constant", value=0.0)
    y0 = pad + dy
    x0 = pad + dx
    return padded[:, :, y0:y0 + h, x0:x0 + w]


def apply_aug(imgs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return imgs
    if mode == "hflip":
        return torch.flip(imgs, dims=[3])
    if mode == "shift_left":
        return shift_tensor_batch(imgs, dx=-2, dy=0)
    if mode == "shift_right":
        return shift_tensor_batch(imgs, dx=2, dy=0)
    if mode == "shift_up":
        return shift_tensor_batch(imgs, dx=0, dy=-2)
    if mode == "shift_down":
        return shift_tensor_batch(imgs, dx=0, dy=2)
    raise ValueError(f"Unknown augmentation mode: {mode}")


@torch.inference_mode()
def true_class_probs(
    model: nn.Module,
    dataset,
    batch_size: int,
    aug_mode: str,
    temperature: float,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_optional_nones,
    )

    out = []
    for batch in loader:
        _, imgs, labels, *_ = unpack_batch(batch)
        imgs = imgs.to(DEVICE)
        labels = None if labels is None else torch.as_tensor(labels, device=DEVICE).long()

        imgs = apply_aug(imgs, aug_mode)
        logits = model(imgs) / float(max(temperature, 1e-6))
        probs = F.softmax(logits, dim=1)
        if labels is None:
            labels = torch.argmax(logits, dim=1)
        p_true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        out.append(p_true.detach().cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(out, axis=0)


def logit_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def train_rmia_reference_models(
    pub_ds,
    cfg: FixedRMIAConfig,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    labels = labels_as_numpy(pub_ds)
    num_classes = int(labels.max()) + 1
    refs = []

    pub_ds.transform = make_eval_transform()

    for r in range(cfg.num_ref_models):
        print(f"\n[RMIA] Training reference model {r+1}/{cfg.num_ref_models}")
        in_idx, _ = stratified_split_indices(
            labels,
            member_frac=cfg.ref_member_frac,
            seed=seed + 4000 + r,
        )
        model = make_model(num_classes=num_classes)
        model = train_one_classifier(
            model,
            pub_ds,
            in_idx,
            epochs=cfg.ref_epoch,
            batch_size=batch_size,
            lr=cfg.ref_lr,
            weight_decay=cfg.ref_weight_decay,
            num_workers=num_workers,
        )
        model.eval()
        refs.append(model)

    return refs


def choose_population_pool(pub_ds, pool_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    members = membership_as_numpy(pub_ds)
    labels = labels_as_numpy(pub_ds)

    candidates = np.where(members == 0)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(pub_ds))

    classes = np.unique(labels[candidates])
    chosen = []
    per_class = max(1, pool_size // max(len(classes), 1))

    for c in classes:
        idx = candidates[labels[candidates] == c]
        rng.shuffle(idx)
        chosen.extend(idx[:per_class].tolist())

    if len(chosen) < pool_size:
        remaining = np.setdiff1d(candidates, np.asarray(chosen, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        chosen.extend(remaining[: pool_size - len(chosen)].tolist())

    chosen = np.asarray(chosen[:pool_size], dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


def compute_augmented_score_cross(
    x_target_probs: np.ndarray,
    x_ref_means: np.ndarray,
    z_target_probs: np.ndarray,
    z_ref_means: np.ndarray,
    pool_idx: np.ndarray,
    gamma: float,
    out_scale: float,
) -> np.ndarray:
    eps = 1e-6
    log_gamma = math.log(max(gamma, 1e-12))

    p_t_x = np.clip(x_target_probs, eps, 1.0 - eps)
    p_r_x = np.clip(x_ref_means * out_scale, eps, 1.0 - eps)
    lr_x = logit_prob(p_t_x) - logit_prob(p_r_x)

    p_t_z = np.clip(z_target_probs[pool_idx], eps, 1.0 - eps)
    p_r_z = np.clip(z_ref_means[pool_idx] * out_scale, eps, 1.0 - eps)
    lr_z = logit_prob(p_t_z) - logit_prob(p_r_z)
    lr_z_sorted = np.sort(lr_z)

    thresholds = lr_x - log_gamma
    score = np.searchsorted(lr_z_sorted, thresholds, side="left") / float(len(lr_z_sorted))
    return score.astype(np.float32)


def aggregate_aug_scores(aug_scores: List[np.ndarray], agg_mode: str) -> np.ndarray:
    stack = np.stack(aug_scores, axis=0)

    if agg_mode == "mean":
        return stack.mean(axis=0).astype(np.float32)
    if agg_mode == "median":
        return np.median(stack, axis=0).astype(np.float32)
    if agg_mode == "majority":
        return (stack >= 0.5).mean(axis=0).astype(np.float32)
    raise ValueError(f"Unknown aggregation mode: {agg_mode}")


def score_rmia_dataset(
    x_target_by_aug: Dict[str, np.ndarray],
    x_ref_mean_by_aug: Dict[str, np.ndarray],
    z_target_by_aug: Dict[str, np.ndarray],
    z_ref_mean_by_aug: Dict[str, np.ndarray],
    pool_idx: np.ndarray,
    gamma: float,
    out_scale: float,
    agg_mode: str,
) -> np.ndarray:
    aug_scores = []
    for aug in x_target_by_aug.keys():
        aug_scores.append(
            compute_augmented_score_cross(
                x_target_probs=x_target_by_aug[aug],
                x_ref_means=x_ref_mean_by_aug[aug],
                z_target_probs=z_target_by_aug[aug],
                z_ref_means=z_ref_mean_by_aug[aug],
                pool_idx=pool_idx,
                gamma=gamma,
                out_scale=out_scale,
            )
        )
    return aggregate_aug_scores(aug_scores, agg_mode=agg_mode)


def run_rmia_attack(
    pub_ds,
    priv_ds,
    target_model,
    cfg: FixedRMIAConfig,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    print("\n" + "=" * 80)
    print(f"[RMIA] Seed {seed}")
    print("=" * 80)

    aug_list = RMIA_AUG_SETS[cfg.aug_mode]
    eval_transform = make_eval_transform()
    pub_ds.transform = eval_transform
    priv_ds.transform = eval_transform
    members_pub = membership_as_numpy(pub_ds).astype(np.int64)

    refs = train_rmia_reference_models(
        pub_ds=pub_ds,
        cfg=cfg,
        seed=seed,
        batch_size=128,
        num_workers=num_workers,
    )

    target_pub = {}
    target_priv = {}
    ref_mean_pub = {aug: np.zeros(len(pub_ds), dtype=np.float32) for aug in aug_list}
    ref_mean_priv = {aug: np.zeros(len(priv_ds), dtype=np.float32) for aug in aug_list}

    print("[RMIA] Scoring target model")
    for aug in aug_list:
        target_pub[aug] = true_class_probs(
            target_model,
            pub_ds,
            batch_size=batch_size,
            aug_mode=aug,
            temperature=cfg.temperature,
            num_workers=num_workers,
        )
        target_priv[aug] = true_class_probs(
            target_model,
            priv_ds,
            batch_size=batch_size,
            aug_mode=aug,
            temperature=cfg.temperature,
            num_workers=num_workers,
        )

    for i, ref_model in enumerate(refs, start=1):
        print(f"[RMIA] Scoring reference model {i}/{len(refs)}")
        for aug in aug_list:
            ref_mean_pub[aug] += true_class_probs(
                ref_model,
                pub_ds,
                batch_size=batch_size,
                aug_mode=aug,
                temperature=cfg.temperature,
                num_workers=num_workers,
            )
            ref_mean_priv[aug] += true_class_probs(
                ref_model,
                priv_ds,
                batch_size=batch_size,
                aug_mode=aug,
                temperature=cfg.temperature,
                num_workers=num_workers,
            )

    for aug in aug_list:
        ref_mean_pub[aug] = np.maximum(ref_mean_pub[aug] / float(cfg.num_ref_models), 1e-6)
        ref_mean_priv[aug] = np.maximum(ref_mean_priv[aug] / float(cfg.num_ref_models), 1e-6)

    pool_idx = choose_population_pool(pub_ds, pool_size=cfg.pool_size, seed=seed)

    pub_scores = score_rmia_dataset(
        x_target_by_aug=target_pub,
        x_ref_mean_by_aug=ref_mean_pub,
        z_target_by_aug=target_pub,
        z_ref_mean_by_aug=ref_mean_pub,
        pool_idx=pool_idx,
        gamma=cfg.gamma,
        out_scale=cfg.out_scale,
        agg_mode=cfg.agg_mode,
    )
    priv_scores = score_rmia_dataset(
        x_target_by_aug=target_priv,
        x_ref_mean_by_aug=ref_mean_priv,
        z_target_by_aug=target_pub,
        z_ref_mean_by_aug=ref_mean_pub,
        pool_idx=pool_idx,
        gamma=cfg.gamma,
        out_scale=cfg.out_scale,
        agg_mode=cfg.agg_mode,
    )

    metrics_pub = evaluate_scores(members_pub, pub_scores)

    del refs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "[RMIA] "
        f"pub TPR@5%FPR={metrics_pub['tpr_at_5fpr']:.4f} | "
        f"AUC={metrics_pub['auc']:.4f}"
    )

    return {
        "pub_scores": pub_scores,
        "priv_scores": priv_scores,
        "pub_metrics": metrics_pub,
    }


# ============================================================
# Ensemble search
# ============================================================
def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def default_weight_grid(step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--weight-step must be > 0")
    steps = int(round(1.0 / step))
    weights = [round(i * step, 10) for i in range(steps + 1)]
    if abs(weights[-1] - 1.0) > 1e-8:
        weights.append(1.0)
    return sorted(set(min(max(w, 0.0), 1.0) for w in weights))


def build_weight_search_space(args) -> List[EnsembleWeightConfig]:
    lira_weights = args.lira_weights or default_weight_grid(args.weight_step)

    if args.rmia_weights is None:
        return [
            EnsembleWeightConfig(lira_weight=float(w), rmia_weight=float(1.0 - w))
            for w in lira_weights
            if 0.0 <= w <= 1.0
        ]

    combos = {}
    for wl, wr in itertools.product(lira_weights, args.rmia_weights):
        if wl == 0.0 and wr == 0.0:
            continue
        if args.normalize_weights:
            total = wl + wr
            wl_eff = float(wl / total)
            wr_eff = float(wr / total)
        else:
            wl_eff = float(wl)
            wr_eff = float(wr)
        combos[(round(wl_eff, 10), round(wr_eff, 10))] = EnsembleWeightConfig(
            lira_weight=wl_eff,
            rmia_weight=wr_eff,
        )

    return sorted(combos.values(), key=lambda cfg: (cfg.lira_weight, cfg.rmia_weight))


def build_seed_list(base_seed: int, num_seeds: int, seed_step: int) -> List[int]:
    if num_seeds <= 0:
        raise ValueError("--num-seeds must be >= 1")
    return [base_seed + i * seed_step for i in range(num_seeds)]


def combine_scores(lira_scores: np.ndarray, rmia_scores: np.ndarray, cfg: EnsembleWeightConfig) -> np.ndarray:
    return (cfg.lira_weight * lira_scores + cfg.rmia_weight * rmia_scores).astype(np.float32)


def run_seed_attacks(
    pub_ds,
    priv_ds,
    target_model,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    set_seed(seed)

    lira = run_lira_attack(
        pub_ds=pub_ds,
        priv_ds=priv_ds,
        target_model=target_model,
        cfg=LIRA_CFG,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    rmia = run_rmia_attack(
        pub_ds=pub_ds,
        priv_ds=priv_ds,
        target_model=target_model,
        cfg=RMIA_CFG,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return {
        "seed": seed,
        "lira_pub_scores": lira["pub_scores"],
        "lira_priv_scores": lira["priv_scores"],
        "rmia_pub_scores": rmia["pub_scores"],
        "rmia_priv_scores": rmia["priv_scores"],
        "lira_pub_metrics": lira["pub_metrics_cal"],
        "rmia_pub_metrics": rmia["pub_metrics"],
    }


def summarize_metric_list(metrics_list: List[Dict[str, float]], name: str) -> Dict[str, float]:
    values = np.asarray([m[name] for m in metrics_list], dtype=np.float32)
    return {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="LiRA + RMIA ensemble weight search")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=97)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--lira-weights", type=parse_csv_floats, default=None)
    parser.add_argument("--rmia-weights", type=parse_csv_floats, default=None)
    parser.add_argument("--normalize-weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_list = build_seed_list(args.seed, args.num_seeds, args.seed_step)
    weight_space = build_weight_search_space(args)

    print(f"Using device: {DEVICE}")
    print(f"Seeds: {seed_list}")
    print(f"Evaluating {len(weight_space)} ensemble weight settings")
    print(f"Fixed LiRA config: {json.dumps(asdict(LIRA_CFG), indent=2)}")
    print(f"Fixed RMIA config: {json.dumps(asdict(RMIA_CFG), indent=2)}")

    pub_ds, priv_ds, target_model = load_datasets_and_model()
    pub_members = membership_as_numpy(pub_ds).astype(np.int64)

    per_seed = []
    for seed in seed_list:
        per_seed.append(
            run_seed_attacks(
                pub_ds=pub_ds,
                priv_ds=priv_ds,
                target_model=target_model,
                seed=seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )

    results = []
    best = None

    with RESULTS_JSONL.open("w", encoding="utf-8") as f:
        for i, weight_cfg in enumerate(weight_space, start=1):
            combined_pub_by_seed = []
            combined_priv_by_seed = []
            seed_metrics = []

            for seed_rec in per_seed:
                pub_scores = combine_scores(
                    seed_rec["lira_pub_scores"],
                    seed_rec["rmia_pub_scores"],
                    weight_cfg,
                )
                priv_scores = combine_scores(
                    seed_rec["lira_priv_scores"],
                    seed_rec["rmia_priv_scores"],
                    weight_cfg,
                )
                combined_pub_by_seed.append(pub_scores)
                combined_priv_by_seed.append(priv_scores)
                seed_metrics.append(evaluate_scores(pub_members, pub_scores))

            avg_pub_scores = np.mean(np.stack(combined_pub_by_seed, axis=0), axis=0).astype(np.float32)
            avg_priv_scores = np.mean(np.stack(combined_priv_by_seed, axis=0), axis=0).astype(np.float32)
            avg_metrics = evaluate_scores(pub_members, avg_pub_scores)

            row = {
                "config_key": weight_cfg.key(),
                "lira_weight": weight_cfg.lira_weight,
                "rmia_weight": weight_cfg.rmia_weight,
                "num_seeds": len(seed_list),
                "score": avg_metrics["tpr_at_5fpr"],
                "auc": avg_metrics["auc"],
                "tpr_at_5fpr": avg_metrics["tpr_at_5fpr"],
                "tpr_at_1fpr": avg_metrics["tpr_at_1fpr"],
                **summarize_metric_list(seed_metrics, "auc"),
                **summarize_metric_list(seed_metrics, "tpr_at_5fpr"),
                **summarize_metric_list(seed_metrics, "tpr_at_1fpr"),
                "seed_metrics": seed_metrics,
            }
            results.append(row)
            f.write(json.dumps(row) + "\n")

            key = (
                avg_metrics["tpr_at_5fpr"],
                avg_metrics["auc"],
                avg_metrics["tpr_at_1fpr"],
            )
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "cfg": weight_cfg,
                    "pub_scores": avg_pub_scores,
                    "priv_scores": avg_priv_scores,
                    "metrics": avg_metrics,
                    "seed_metrics": seed_metrics,
                }
                print(
                    f"[Ensemble] New best {i}/{len(weight_space)} | "
                    f"lira={weight_cfg.lira_weight:.3f}, rmia={weight_cfg.rmia_weight:.3f} | "
                    f"TPR@5%FPR={avg_metrics['tpr_at_5fpr']:.4f} | "
                    f"AUC={avg_metrics['auc']:.4f}"
                )

    results_df = pd.DataFrame(results).sort_values(
        by=["tpr_at_5fpr", "auc", "tpr_at_1fpr"],
        ascending=False,
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    if best is None:
        raise RuntimeError("No ensemble configurations were evaluated.")

    best_cfg = best["cfg"]
    with BEST_CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": asdict(best_cfg),
                "metrics": best["metrics"],
                "seed_metrics": best["seed_metrics"],
                "seeds": seed_list,
                "lira_config": asdict(LIRA_CFG),
                "rmia_config": asdict(RMIA_CFG),
            },
            f,
            indent=2,
        )

    priv_ids = [str(x) for x in priv_ds.ids]
    submission_df = pd.DataFrame({
        "id": priv_ids,
        "score": best["priv_scores"],
    })
    submission_df.to_csv(OUTPUT_CSV, index=False)

    print("\nBest ensemble:")
    print(f"  LiRA weight: {best_cfg.lira_weight:.4f}")
    print(f"  RMIA weight: {best_cfg.rmia_weight:.4f}")
    print(f"  Pub TPR@5%FPR: {best['metrics']['tpr_at_5fpr']:.4f}")
    print(f"  Pub AUC: {best['metrics']['auc']:.4f}")
    print("\nSaved:")
    print(f"  {RESULTS_JSONL}")
    print(f"  {RESULTS_CSV}")
    print(f"  {BEST_CONFIG_JSON}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
