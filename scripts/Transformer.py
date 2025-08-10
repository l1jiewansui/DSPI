import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def _load_csv_matrix(data_dir: Path) -> np.ndarray:
    preferred = data_dir / "skewed_dataset.csv"
    if preferred.exists():
        df = pd.read_csv(preferred)
    else:
        csvs = list(data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {data_dir}")
        df = pd.read_csv(csvs[0])
    arr = df.values.astype(float)
    return arr


def _normalize(x: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)


def _select_with_farthest_first(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    idxs = []
    first = np.random.randint(n)
    idxs.append(first)
    dists = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, k):
        cand = int(np.argmax(dists))
        idxs.append(cand)
        dists = np.minimum(dists, np.linalg.norm(points - points[cand], axis=1))
    return np.array(idxs, dtype=int)


def _assign_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # x: [N,D], centers: [K,D]
    # return cluster ids [N]
    # Use batch distance
    dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dists, axis=1)


class _TinyTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 64, nhead: int = 4, ff: int = 128, nlayers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, D]
        z = self.proj(x)
        z = self.enc(z.unsqueeze(0)).squeeze(0)  # [S, d_model]
        s = self.head(z).squeeze(-1)  # [S]
        return s


def _score_with_transformer(x: np.ndarray, sample_size: int = 8192, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    xs = torch.from_numpy(x[idx]).float().to(device)
    model = _TinyTransformer(d_in=d, d_model=min(64, max(16, d * 2)), nhead=4, ff=128, nlayers=2).to(device)
    model.eval()
    with torch.no_grad():
        scores = model(xs).cpu().numpy()
    return idx, scores


def _save_outputs(out_dir: Path, normalized: np.ndarray, clusters: np.ndarray, main_pivots: np.ndarray, per_cluster_pivots: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = main_pivots.shape[0]
    for i in range(K):
        mask = clusters == i
        cluster_data = normalized[mask]
        ids = np.arange(cluster_data.shape[0]).reshape(-1, 1)
        mat = np.hstack([ids, cluster_data])
        np.save(out_dir / f"cluster_{i}.npy", mat)
        fmt = "%i," + ",".join(["%.6f"] * cluster_data.shape[1])
        np.savetxt(out_dir / f"cluster_{i}.txt", mat, fmt=fmt, delimiter=",")
        # per-cluster additional pivots
        piv = per_cluster_pivots.get(i, np.empty((0, cluster_data.shape[1])))
        if piv.size:
            np.savetxt(out_dir / f"pivot_{i}.txt", piv, fmt=",".join(["%.6f"] * piv.shape[1]), delimiter=",")
    np.savetxt(out_dir / "ref.txt", main_pivots, fmt=",".join(["%.6f"] * main_pivots.shape[1]), delimiter=",")


def gen_eqdata(data_dir: str, out_dir: str, num_clusters: int):
    """
    Generate clusters and pivots using a lightweight Transformer-based pivot scoring.
    - data_dir: directory with a CSV (prefers 'skewed_dataset.csv')
    - out_dir: output directory to write cluster_*.txt, pivot_*.txt and ref.txt
    - num_clusters: number of main pivots (clusters)
    """
    t0 = time.time()
    data_dir_p = Path(data_dir)
    out_dir_p = Path(out_dir)

    x = _load_csv_matrix(data_dir_p)
    x = x.astype(float)
    if x.ndim != 2:
        raise ValueError("Input CSV must be 2D array-like")

    x = _normalize(x)

    # Transformer-based scoring on a subset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx_sub, scores = _score_with_transformer(x, sample_size=min(8192, max(1024, num_clusters * 200))),
    # The line above returns a tuple inside a tuple if comma used; fix:
    idx_sub, scores = idx_sub  # type: ignore
    # select candidate centers from subset using farthest-first on top-scored candidates
    top_k = min(len(idx_sub), max(num_clusters * 5, num_clusters))
    top_idx_local = np.argsort(scores)[-top_k:]
    cand = x[idx_sub[top_idx_local]]
    main_idx_local = _select_with_farthest_first(cand, k=num_clusters)
    main_pivots = cand[main_idx_local]

    # assign all points to nearest main pivot
    clusters = _assign_to_centers(x, main_pivots)

    # per-cluster two additional pivots via farthest-first within cluster
    per_cluster_pivots = {}
    for i in range(num_clusters):
        pts = x[clusters == i]
        if pts.shape[0] == 0:
            continue
        m = min(2, max(0, pts.shape[0]))
        if m == 0:
            continue
        if pts.shape[0] <= m:
            piv = pts
        else:
            idxs = _select_with_farthest_first(pts, k=m)
            piv = pts[idxs]
        per_cluster_pivots[i] = piv

    _save_outputs(out_dir_p, x, clusters, main_pivots, per_cluster_pivots)
    print(f"Transformer builder done. Output: {out_dir}  time={time.time()-t0:.2f}s")