"""Local re-implementations of the UPAR metrics, written from the formulas on the Codabench Evaluation pages
(read 2026-10-04). The organizers' scoring programs are not public, so these are NOT guaranteed to match them to the
last digit (empty-set handling and epsilons are my choice); use them to compare your own runs, not to predict a rank.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def par_metrics(probs: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    """Track 1. probs, gt: (N, 40). Present = prob above thr. Challenge_Avg = harmonic mean of mA and Instance_F1."""
    p = (np.asarray(probs) > thr).astype(np.float64)
    g = (np.asarray(gt) > 0.5).astype(np.float64)
    tp, tn = (p * g).sum(0), ((1 - p) * (1 - g)).sum(0)
    fp, fn = (p * (1 - g)).sum(0), ((1 - p) * g).sum(0)
    pos_acc = tp / np.maximum(tp + fn, EPS)
    neg_acc = tn / np.maximum(tn + fp, EPS)
    valid = (tp + fn > 0) & (tn + fp > 0)  # an attribute with no positives (or no negatives) has no defined mA term
    mA = float(((pos_acc + neg_acc) / 2)[valid].mean()) if valid.any() else 0.0
    label_f1 = float((2 * tp / np.maximum(2 * tp + fp + fn, EPS)).mean())
    inter = (p * g).sum(1)
    union = np.maximum(p, g).sum(1)
    ip = inter / np.maximum(p.sum(1), EPS)
    ir = inter / np.maximum(g.sum(1), EPS)
    inst_f1 = float((2 * ip * ir / np.maximum(ip + ir, EPS)).mean())
    return {
        "mA": mA, "Label_F1": label_f1, "Instance_Acc": float((inter / np.maximum(union, EPS)).mean()),
        "Instance_Prec": float(ip.mean()), "Instance_Rec": float(ir.mean()), "Instance_F1": inst_f1,
        "Challenge_Avg": float(2 * mA * inst_f1 / max(mA + inst_f1, EPS)),
    }


def retrieval_metrics(dist: np.ndarray, queries: np.ndarray, gallery_gt: np.ndarray) -> dict:
    """Track 2. dist: (Q, G), smaller = better; equal values keep gallery order. A gallery image is a correct match for
    a query only if all 40 attributes equal the query. Queries with no correct match are skipped.
    mADM is the ranking metric (page formulas: AP, INP, ADM with per-query normalised degree of match)."""
    dist = np.asarray(dist, dtype=np.float64)
    q = (np.asarray(queries) > 0.5).astype(np.int8)
    g = (np.asarray(gallery_gt) > 0.5).astype(np.int8)
    ap, inp, adm, ranks = [], [], [], []
    for i in range(dist.shape[0]):
        agree = (g == q[i]).sum(1)  # (G,) attributes equal to the query
        correct = agree == q.shape[1]
        R = int(correct.sum())
        if R == 0:
            continue
        order = np.argsort(dist[i], kind="stable")
        r = np.flatnonzero(correct[order]) + 1  # 1-based ranks of the correct matches, ascending
        k = np.arange(1, R + 1)
        ap.append(float((k / r).mean()))
        inp.append(float(R / r[-1]))
        ranks.append(int(r[0]))
        dom = agree / q.shape[1]
        denom = dom.max() - dom.mean()
        ndom = np.maximum(0.0, (dom - dom.mean()) / denom) if denom > 0 else np.zeros_like(dom)
        csum = np.cumsum(ndom[order])
        adm.append(float((csum[r - 1] / r).mean()))
    ranks = np.array(ranks)
    return {
        "mADM": float(np.mean(adm)), "mAP": float(np.mean(ap)), "mINP": float(np.mean(inp)),
        "Rank-1": float((ranks <= 1).mean()), "Rank-5": float((ranks <= 5).mean()), "Rank-10": float((ranks <= 10).mean()),
        "queries_scored": len(ap),
    }


def distances_l1(probs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The organizers' recipe: sum|q - p| = sum(p) + q . (1 - 2p), without a (Q, G, 40) tensor."""
    probs = np.asarray(probs, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    return probs.sum(1)[None, :] + queries @ (1.0 - 2.0 * probs).T


def distances_nll(probs: np.ndarray, queries: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Negative log-likelihood of the query under independent attribute probabilities (smaller = better match)."""
    p = np.clip(np.asarray(probs, dtype=np.float32), eps, 1 - eps)
    q = np.asarray(queries, dtype=np.float32)
    return -(q @ np.log(p).T + (1.0 - q) @ np.log1p(-p).T)
