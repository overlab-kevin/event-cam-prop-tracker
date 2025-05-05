#!/usr/bin/env python
"""
Train the miniature SpConv UNet on labelled event batches.
Call from repo root:

    python main.py                # fresh run
    python main.py --resume ckpt  # resume from checkpoint

TensorBoard logdir = cfg.output_dir
"""
from pathlib import Path
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from config.config import Cfg
from src.dataloader import make_loaders
from src.model import SpMiniUNetWrapper


# ───────────────────────── helpers ──────────────────────────────
def bce_loss(pred_logits, target):
    """
    pred_logits : (N,) or (N,1)  – raw scores from the network
    target      : (N,)           – 0 / 1 voxel labels
    """
    if pred_logits.ndim == 2:            # (N,1) → (N,)
        pred_logits = pred_logits.squeeze(1)

    n_pos = target.sum().clamp(min=1.)    # avoid /0
    n_neg = target.numel() - n_pos
    pos_weight = n_neg / n_pos

    return torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logits,
        target.float(),
        pos_weight=pos_weight,
    )


def train_one_epoch(model, loader, opt, tb, epoch):
    model.train()
    total = 0.0
    for i, batch in enumerate(loader):
        opt.zero_grad()
        feats_out = model.forward_batch(batch)            # list-of-lists

        logits_list, target_list = [], []
        for item_feats, item in zip(feats_out, batch):
            for g_idx, grid_feats in enumerate(item_feats):
                logits_list.append(grid_feats.squeeze(1))        # N×1 → N
                target_list.append(item["target"][g_idx].to(grid_feats.device))

        logits  = torch.cat(logits_list, 0)   # (total_voxels,)
        target  = torch.cat(target_list, 0)   # (total_voxels,)
        loss = bce_loss(logits, target)
        loss.backward()
        opt.step()

        step = epoch * len(loader) + i
        tb.add_scalar("train/loss", loss.item(), step)
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, tb, epoch, thresh_logit: float = 0.0):
    """
    Evaluate on validation split and push metrics to TensorBoard.

    thresh_logit : decision threshold in logit space (0.0 ≙ prob 0.5)
    """
    model.eval()

    # running sums
    loss_sum = 0.0
    tp = fp = fn = tn = 0

    for batch in loader:
        feats_out = model.forward_batch(batch)

        logits_list, target_list = [], []
        for item_feats, item in zip(feats_out, batch):
            for g_idx, grid_feats in enumerate(item_feats):
                logits_list.append(grid_feats.squeeze(1))                  # (Ni,)
                target_list.append(item["target"][g_idx].to(grid_feats.device))

        logits = torch.cat(logits_list, 0)          # raw scores
        target = torch.cat(target_list, 0)          # 0 / 1

        # 1. BCE loss (already pos-weighted)
        loss = bce_loss(logits, target)
        loss_sum += loss.item()

        # 2. threshold and gather confusion counts
        pred = (logits >= thresh_logit).to(torch.uint8)
        tp += int(((pred == 1) & (target == 1)).sum())
        fp += int(((pred == 1) & (target == 0)).sum())
        fn += int(((pred == 0) & (target == 1)).sum())
        tn += int(((pred == 0) & (target == 0)).sum())

    # ─── derive metrics ─────────────────────────────────────────
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    avg_loss  = loss_sum / len(loader)

    # ─── tensorboard ────────────────────────────────────────────
    tb.add_scalar("val/loss",      avg_loss,  epoch)
    tb.add_scalar("val/precision", precision, epoch)
    tb.add_scalar("val/recall",    recall,    epoch)
    tb.add_scalar("val/f1",        f1,        epoch)
    tb.add_scalar("val/iou",       iou,       epoch)
    tb.add_scalar("val/accuracy",  accuracy,  epoch)

    print(
        f"[val] epoch {epoch:03d} "
        f"loss {avg_loss:.4f}  "
        f"P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  IoU {iou:.3f}"
    )
    return avg_loss


# ─────────────────────────── main ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, help="checkpoint path to resume")
    args = ap.parse_args()

    cfg = Cfg()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir)

    tb = SummaryWriter(out_dir)

    train_loader, val_loader = make_loaders(cfg)

    model = SpMiniUNetWrapper(cfg).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    best = float("inf")
    for epoch in range(start_epoch, cfg.train.epochs):
        tr_loss = train_one_epoch(model, train_loader, opt, tb, epoch)
        val_loss = val_epoch(model, val_loader, tb, epoch)
        print(f"Epoch {epoch:03d}  train {tr_loss:.4f}  val {val_loss:.4f}")

        # save latest
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()},
            out_dir / "latest.pt",
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()},
                out_dir / "best.pt",
            )
