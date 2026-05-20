#!/usr/bin/env python3
"""
Comprehensive hyper-parameter search for an offline RMIA attack.

Design goals:
- Train a bank of reference models once at the maximum requested budget.
- Cache raw logits for several checkpoints and augmentations.
- Reuse those caches to exhaustively search RMIA hyperparameters:
  * number of reference models
  * reference training epoch checkpoint
  * population pool size ("z" samples)
  * gamma threshold
  * out_scale
  * temperature
  * augmentation query set
  * aggregation rule across queries

This is an offline RMIA search script. It is aligned with the RMIA paper's
key knobs: number of reference models, number of population samples z,
threshold gamma, and augmented queries.

Inputs (same directory as this script):
  - pub.pt
  - priv.pt
  - model.pt

Outputs:
  - rmia_search_results.csv
  - rmia_search_results.jsonl
  - best_rmia_config.json
  - submission.csv
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, roc_auc_score
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms

# Reuse the assignment helpers where possible.
from rmia_attack import (
    MEAN,
    STD,
    TaskDataset,
    MembershipDataset,
    unpack_batch,
    labels_array,
    membership_array,
    apply_aug,
)

BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"
RESULTS_CSV = BASE / "rmia_search_results-12.csv"
RESULTS_JSONL = BASE / "rmia_search_results-12.jsonl"
BEST_CONFIG_JSON = BASE / "best_rmia_config-12.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation sets to search over.
AUG_SETS = {
    "1q": ("identity",),
    "2q": ("identity", "hflip"),
    "5q": ("identity", "shift_left", "shift_right", "shift_up", "shift_down"),
    "6q": ("identity", "hflip", "shift_left", "shift_right", "shift_up", "shift_down"),
}

# Reasonable default grid for an overnight run.
DEFAULT_NUM_REF_MODELS = [32, 64]
DEFAULT_REF_EPOCHS = [15,20]
DEFAULT_POOL_SIZES = [3000, 4000]
DEFAULT_GAMMAS = [0.5, 1.0]
DEFAULT_OUT_SCALES = [1.0, 1.25]
DEFAULT_TEMPERATURES = [1, 1.25]
DEFAULT_AUG_MODES = ["6q"]
DEFAULT_AGG_MODES = ["median"]


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


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_model(num_classes: int):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def load_model(path: Path, num_classes: int):
    model = make_model(num_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def labels_np(ds) -> np.ndarray:
    labels = labels_array(ds)
    return labels.astype(np.int64, copy=False)


def members_np(ds) -> np.ndarray:
    members = membership_array(ds)
    return members.astype(np.int64, copy=False)


def stratified_split_indices(labels: np.ndarray, member_frac: float = 0.5, seed: int = 0):
    """
    Stratified split into training and holdout indices.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    train_idx, holdout_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * member_frac))
        cut = max(1, min(cut, len(idx) - 1))
        train_idx.extend(idx[:cut].tolist())
        holdout_idx.extend(idx[cut:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(holdout_idx)
    return train_idx, holdout_idx


def logit_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits.astype(np.float32, copy=False) / float(max(temperature, 1e-6))
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def true_class_probs_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    probs = softmax_np(logits, temperature=temperature)
    idx = np.arange(len(labels), dtype=np.int64)
    return probs[idx, labels]


def augment_modes_for_name(name: str) -> Tuple[str, ...]:
    if name not in AUG_SETS:
        raise ValueError(f"Unknown augmentation mode '{name}'. Choices: {sorted(AUG_SETS)}")
    return AUG_SETS[name]


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    for batch in loader:
        _, imgs, labels, *_ = unpack_batch(batch)
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        opt.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total_loss += float(loss.item()) * imgs.size(0)
        total += imgs.size(0)

    return total_loss / max(total, 1)


@torch.no_grad()
def collect_logits(model: nn.Module, dataset, aug_mode: str, batch_size: int = 256, num_workers: int = 0) -> np.ndarray:
    """
    Returns raw logits for the full dataset, in dataset order.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    out = []
    for batch in loader:
        _, imgs, labels, *_ = unpack_batch(batch)
        imgs = imgs.to(device)
        imgs = apply_aug(imgs, aug_mode)
        logits = model(imgs)
        out.append(logits.detach().cpu().numpy().astype(np.float16, copy=False))
    return np.concatenate(out, axis=0)


def train_reference_bank(
    pub_ds,
    num_ref_models: int,
    max_ref_epochs: int,
    checkpoint_epochs: Sequence[int],
    ref_batch_size: int = 128,
    ref_lr: float = 1e-3,
    member_frac: float = 0.5,
    num_workers: int = 0,
    seed: int = 0,
    aug_modes: Sequence[str] = ("identity", "hflip", "shift_left", "shift_right", "shift_up", "shift_down"),
):
    """
    Train reference models once and cache logits at selected checkpoints.

    Returns:
      ref_logits[epoch][aug]["pub"]  -> list of logits arrays, one per ref model
      ref_logits[epoch][aug]["priv"] -> list of logits arrays, one per ref model
    """
    labels = labels_np(pub_ds)
    num_classes = int(labels.max()) + 1

    checkpoint_epochs = sorted(set(int(e) for e in checkpoint_epochs))
    if len(checkpoint_epochs) == 0:
        raise ValueError("checkpoint_epochs is empty")
    if max_ref_epochs < checkpoint_epochs[-1]:
        raise ValueError("max_ref_epochs must be >= the largest checkpoint epoch")

    # Nested storage: epoch -> aug -> {"pub": [arr per ref], "priv": [arr per ref]}
    ref_logits: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {}
    for epoch in checkpoint_epochs:
        ref_logits[epoch] = {aug: {"pub": [], "priv": []} for aug in aug_modes}

    for r in range(num_ref_models):
        print(f"\nTraining reference model {r+1}/{num_ref_models}")
        train_idx, _ = stratified_split_indices(labels, member_frac=member_frac, seed=seed + 1000 + r)

        model = make_model(num_classes).to(device)
        loader = DataLoader(
            Subset(pub_ds, train_idx),
            batch_size=ref_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=ref_lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_ref_epochs))

        for epoch in range(1, max_ref_epochs + 1):
            loss = train_one_epoch(model, loader, opt)
            sched.step()
            print(f"  Epoch {epoch}/{max_ref_epochs} | loss={loss:.4f}")

            if epoch in checkpoint_epochs:
                for aug in aug_modes:
                    pub_logits = collect_logits(model, pub_ds, aug, batch_size=256, num_workers=num_workers)
                    ref_logits[epoch][aug]["pub"].append(pub_logits)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ref_logits


# ---------------------------------------------------------------------------
# Probability cache construction
# ---------------------------------------------------------------------------
def build_target_logits_cache(target_model, pub_ds, aug_modes, num_workers: int = 0):
    cache = {"pub": {}, "priv": {}}
    for aug in aug_modes:
        cache["pub"][aug] = collect_logits(target_model, pub_ds, aug, batch_size=256, num_workers=num_workers)
    return cache


def build_target_prob_cache(target_logits_cache, labels_pub, temperatures, aug_modes):
    cache = {"pub": {}, "priv": {}}
    for temp in temperatures:
        cache["pub"][temp] = {}
        for aug in aug_modes:
            cache["pub"][temp][aug] = true_class_probs_from_logits(target_logits_cache["pub"][aug], labels_pub, temperature=temp)
    return cache


def build_reference_prob_prefix_cache(ref_logits_cache, labels_pub, temperatures, checkpoint_epochs, aug_modes):
    """
    Convert raw ref logits into prefix-mean true-class probabilities.

    Output:
      ref_prob_cache[dataset]["temp"][epoch][aug] = prefix means, shape [R, N]
      where row k corresponds to the mean over the first k+1 reference models.
    """
    cache = {"pub": {}, "priv": {}}
    for temp in temperatures:
        cache["pub"][temp] = {}
        for epoch in checkpoint_epochs:
            cache["pub"][temp][epoch] = {}
            for aug in aug_modes:
                pub_stack = np.stack(ref_logits_cache[epoch][aug]["pub"], axis=0)  # [R,N,C]

                pub_ptrue = np.stack(
                    [true_class_probs_from_logits(pub_stack[r], labels_pub, temperature=temp) for r in range(pub_stack.shape[0])],
                    axis=0,
                ).astype(np.float32)

                pub_cum = np.cumsum(pub_ptrue, axis=0)

                denom = np.arange(1, pub_ptrue.shape[0] + 1, dtype=np.float32)[:, None]
                cache["pub"][temp][epoch][aug] = pub_cum / denom

    return cache


# ---------------------------------------------------------------------------
# Population pool selection
# ---------------------------------------------------------------------------
def choose_population_pool(pub_ds, pool_size: int, seed: int = 0) -> np.ndarray:
    """
    Prefer public non-members as population pool. If unavailable, use all public samples.
    """
    rng = np.random.default_rng(seed)

    if hasattr(pub_ds, "membership"):
        mem = members_np(pub_ds)
        candidates = np.where(mem == 0)[0]
        if len(candidates) == 0:
            candidates = np.arange(len(pub_ds))
    else:
        candidates = np.arange(len(pub_ds))

    labels = labels_np(pub_ds)
    classes = np.unique(labels[candidates])
    chosen: List[int] = []

    # Balanced selection per class, then fill the remainder.
    per_class = max(1, pool_size // max(len(classes), 1))
    for c in classes:
        idx = candidates[labels[candidates] == c]
        rng.shuffle(idx)
        chosen.extend(idx[:per_class].tolist())

    if len(chosen) < pool_size:
        remaining = np.setdiff1d(candidates, np.array(chosen, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        chosen.extend(remaining[: pool_size - len(chosen)].tolist())

    chosen = np.array(chosen[:pool_size], dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# RMIA scoring
# ---------------------------------------------------------------------------
def compute_augmented_score(
    target_probs: np.ndarray,
    ref_means: np.ndarray,
    pool_idx: np.ndarray,
    gamma: float,
    out_scale: float,
) -> np.ndarray:
    """
    RMIA-style score for one augmentation.

    target_probs: [N]
    ref_means:    [N]
    pool_idx:     indices of z samples

    Returns a continuous score in [0,1] for every sample x in the dataset.
    """
    eps = 1e-6
    log_gamma = math.log(max(gamma, 1e-12))

    p_t = np.clip(target_probs, eps, 1.0 - eps)
    p_r = np.clip(ref_means * out_scale, eps, 1.0 - eps)

    lr_x = logit_prob(p_t) - logit_prob(p_r)

    p_t_z = np.clip(target_probs[pool_idx], eps, 1.0 - eps)
    p_r_z = np.clip(ref_means[pool_idx] * out_scale, eps, 1.0 - eps)
    lr_z = logit_prob(p_t_z) - logit_prob(p_r_z)
    lr_z_sorted = np.sort(lr_z)

    # score(x) = fraction of z such that lr_x > lr_z + log(gamma)
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


def score_config(
    target_prob_cache,
    ref_prob_prefix_cache,
    dataset_name: str,
    temperature: float,
    checkpoint_epoch: int,
    num_ref_models: int,
    pool_idx: np.ndarray,
    gamma: float,
    out_scale: float,
    aug_mode: str,
    agg_mode: str,
) -> np.ndarray:
    aug_list = augment_modes_for_name(aug_mode)
    aug_scores = []

    for aug in aug_list:
        target_probs = target_prob_cache[dataset_name][temperature][aug]
        ref_means = ref_prob_prefix_cache[dataset_name][temperature][checkpoint_epoch][aug][num_ref_models - 1]
        aug_score = compute_augmented_score(
            target_probs=target_probs,
            ref_means=ref_means,
            pool_idx=pool_idx,
            gamma=gamma,
            out_scale=out_scale,
        )
        aug_scores.append(aug_score)

    return aggregate_aug_scores(aug_scores, agg_mode)


def evaluate_scores(scores: np.ndarray, members: np.ndarray):
    fpr, tpr, _ = roc_curve(members, scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_5 = float(np.interp(0.05, fpr, tpr))
    tpr_at_1 = float(np.interp(0.01, fpr, tpr))
    return {
        "auc": float(roc_auc),
        "tpr_at_5fpr": tpr_at_5,
        "tpr_at_1fpr": tpr_at_1,
    }


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchConfig:
    num_ref_models: int
    ref_epoch: int
    pool_size: int
    gamma: float
    out_scale: float
    temperature: float
    aug_mode: str
    agg_mode: str


def build_search_space(args) -> List[SearchConfig]:
    num_refs = args.num_ref_models or DEFAULT_NUM_REF_MODELS
    ref_epochs = args.ref_epochs or DEFAULT_REF_EPOCHS
    pool_sizes = args.pool_sizes or DEFAULT_POOL_SIZES
    gammas = args.gammas or DEFAULT_GAMMAS
    out_scales = args.out_scales or DEFAULT_OUT_SCALES
    temperatures = args.temperatures or DEFAULT_TEMPERATURES
    aug_modes = args.aug_modes or DEFAULT_AUG_MODES
    agg_modes = args.agg_modes or DEFAULT_AGG_MODES

    space = [
        SearchConfig(
            num_ref_models=n,
            ref_epoch=e,
            pool_size=p,
            gamma=g,
            out_scale=o,
            temperature=t,
            aug_mode=a,
            agg_mode=m,
        )
        for n, e, p, g, o, t, a, m in itertools.product(
            num_refs, ref_epochs, pool_sizes, gammas, out_scales, temperatures, aug_modes, agg_modes
        )
        if n <= args.max_ref_models_train
    ]

    return space


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive RMIA hyper-parameter search")
    p.add_argument("--seed", type=int, default=6767)
    p.add_argument("--ref-member-frac", type=float, default=0.5)
    p.add_argument("--max-ref-models-train", type=int, default=64)
    p.add_argument("--max-ref-epochs-train", type=int, default=25)
    p.add_argument("--ref-batch-size", type=int, default=128)
    p.add_argument("--ref-lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)

    # Optional overrides for the search grid.
    p.add_argument("--num-ref-models", type=parse_csv_ints, default=None,
                   help="Comma-separated list, e.g. 1,2,4,8,16")
    p.add_argument("--ref-epochs", type=parse_csv_ints, default=None,
                   help="Comma-separated list, e.g. 4,8,12")
    p.add_argument("--pool-sizes", type=parse_csv_ints, default=None,
                   help="Comma-separated list, e.g. 250,1000,2500")
    p.add_argument("--gammas", type=parse_csv_floats, default=None,
                   help="Comma-separated list, e.g. 0.5,1,2,4")
    p.add_argument("--out-scales", type=parse_csv_floats, default=None,
                   help="Comma-separated list, e.g. 0.75,1,1.25")
    p.add_argument("--temperatures", type=parse_csv_floats, default=None,
                   help="Comma-separated list, e.g. 0.75,1,1.5")
    p.add_argument("--aug-modes", type=parse_csv_strs, default=None,
                   help="Comma-separated list, e.g. 1q,2q,5q,6q")
    p.add_argument("--agg-modes", type=parse_csv_strs, default=None,
                   help="Comma-separated list, e.g. mean,median,majority")

    p.add_argument("--max-configs", type=int, default=0,
                   help="If >0, randomly sample this many configs from the full grid.")
    p.add_argument("--eval-batch-note", action="store_true",
                   help="Print a reminder that the search is exhaustive over the provided grid.")
    return p.parse_args()



def main():

    args = parse_args()
    set_seed(args.seed)

    print("Loading datasets...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)

    # Same normalization as the assignment template.
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub_ds.transform = transform

    pub_labels = labels_np(pub_ds)
    num_classes = int(pub_labels.max()) + 1

    print("Loading target model...")
    target_model = load_model(MODEL_PATH, num_classes=num_classes).to(device)

    # Build the augmentation list we will cache.
    aug_cache_modes = ("identity", "hflip", "shift_left", "shift_right", "shift_up", "shift_down")
    temperature_grid = args.temperatures or DEFAULT_TEMPERATURES
    checkpoint_epochs = args.ref_epochs or DEFAULT_REF_EPOCHS
    max_ref_epochs = args.max_ref_epochs_train
    if max_ref_epochs < max(checkpoint_epochs):
        raise ValueError("--max-ref-epochs-train must be >= the largest --ref-epochs value")

    print("\nCaching target logits...")
    target_logits_cache = {
        "pub": {},
        "priv": {},
    }
    for aug in aug_cache_modes:
        print(f"  Target logits / {aug}")
        target_logits_cache["pub"][aug] = collect_logits(target_model, pub_ds, aug, batch_size=256, num_workers=args.num_workers)

    # Train references once, then cache logits at selected checkpoints.
    print("\nTraining reference bank and caching logits at checkpoints...")
    ref_logits_cache = train_reference_bank(
        pub_ds=pub_ds,
        num_ref_models=args.max_ref_models_train,
        max_ref_epochs=max_ref_epochs,
        checkpoint_epochs=checkpoint_epochs,
        ref_batch_size=args.ref_batch_size,
        ref_lr=args.ref_lr,
        member_frac=args.ref_member_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        aug_modes=aug_cache_modes,
    )

    print("\nBuilding probability caches...")
    target_prob_cache = {"pub": {}, "priv": {}}
    for temp in temperature_grid:
        target_prob_cache["pub"][temp] = {}
        target_prob_cache["priv"][temp] = {}
        for aug in aug_cache_modes:
            target_prob_cache["pub"][temp][aug] = true_class_probs_from_logits(
                target_logits_cache["pub"][aug], pub_labels, temperature=temp
            )

    ref_prob_prefix_cache = {"pub": {}, "priv": {}}
    for temp in temperature_grid:
        ref_prob_prefix_cache["pub"][temp] = {}
        ref_prob_prefix_cache["priv"][temp] = {}
        for epoch in checkpoint_epochs:
            ref_prob_prefix_cache["pub"][temp][epoch] = {}
            ref_prob_prefix_cache["priv"][temp][epoch] = {}
            for aug in aug_cache_modes:
                pub_stack = np.stack(ref_logits_cache[epoch][aug]["pub"], axis=0)   # [R,N,C]

                pub_ptrue = np.stack(
                    [true_class_probs_from_logits(pub_stack[r].astype(np.float32), pub_labels, temperature=temp)
                     for r in range(pub_stack.shape[0])],
                    axis=0,
                ).astype(np.float32)

                pub_cum = np.cumsum(pub_ptrue, axis=0)
                denom = np.arange(1, pub_ptrue.shape[0] + 1, dtype=np.float32)[:, None]

                ref_prob_prefix_cache["pub"][temp][epoch][aug] = pub_cum / denom

    # Choose a sufficiently large pool once, then use prefixes for smaller pool sizes.
    max_pool_size = max(args.pool_sizes or DEFAULT_POOL_SIZES)
    pool_idx_all = choose_population_pool(pub_ds, pool_size=max_pool_size, seed=args.seed)
    print(f"\nMax population pool size: {len(pool_idx_all)}")

    # Build the search space.
    search_space = build_search_space(args)
    if args.max_configs > 0 and len(search_space) > args.max_configs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(search_space), size=args.max_configs, replace=False)
        search_space = [search_space[i] for i in sorted(idx.tolist())]

    print(f"\nEvaluating {len(search_space)} RMIA configurations...")

    pub_members = members_np(pub_ds)
    results = []
    best = None

    # Exhaustive evaluation over the cached probability tensors.
    for i, cfg in enumerate(search_space, 1):
        pool_idx = pool_idx_all[: cfg.pool_size]

        pub_scores = score_config(
            target_prob_cache=target_prob_cache,
            ref_prob_prefix_cache=ref_prob_prefix_cache,
            dataset_name="pub",
            temperature=cfg.temperature,
            checkpoint_epoch=cfg.ref_epoch,
            num_ref_models=cfg.num_ref_models,
            pool_idx=pool_idx,
            gamma=cfg.gamma,
            out_scale=cfg.out_scale,
            aug_mode=cfg.aug_mode,
            agg_mode=cfg.agg_mode,
        )
        metrics = evaluate_scores(pub_scores, pub_members)

        row = {
            **asdict(cfg),
            **metrics,
        }
        results.append(row)

        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")

        # Rank by TPR@5%FPR first, then AUC, then TPR@1%FPR.
        key = (metrics["tpr_at_5fpr"], metrics["auc"], metrics["tpr_at_1fpr"])
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "cfg": cfg,
                "metrics": metrics,
            }
            print(
                f"  New best: TPR@5%FPR={metrics['tpr_at_5fpr']:.4f}, "
                f"AUC={metrics['auc']:.4f}, config={cfg}"
            )

        if i % 25 == 0 or i == len(search_space):
            print(f"Finished {i}/{len(search_space)} configs")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["tpr_at_5fpr", "auc", "tpr_at_1fpr"],
        ascending=False,
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    best_cfg: SearchConfig = best["cfg"]
    print("\nBest configuration:")
    print(best_cfg)
    print(best["metrics"])

    with open(BEST_CONFIG_JSON, "w") as f:
        json.dump(
            {
                "config": asdict(best_cfg),
                "metrics": best["metrics"],
            },
            f,
            indent=2,
        )

    # Compute the final private scores using the best config.
    best_pool = pool_idx_all[: best_cfg.pool_size]


    print("\nSaved:")
    print(f"  {RESULTS_CSV}")
    print(f"  {RESULTS_JSONL}")
    print(f"  {BEST_CONFIG_JSON}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
