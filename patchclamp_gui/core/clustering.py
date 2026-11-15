"""无监督聚类的事件动力学分析。

两种表征：
* "features"：幅度、上升时间、衰减 τ、半宽、电荷等标量特征（偏态量先取对数再标准化）；
* "waveform"：峰值归一化后的事件波形做 PCA —— 只看形状、与幅度无关，最适合回答
  "是否存在动力学不同的事件亚群"（例如近端/远端输入、不同亚基组成的受体、不同中间神经元来源）。

聚类前先检验"是否真的存在多簇"：比较 GMM 在 k=1 与 k>1 时的 BIC，并给出轮廓系数。
连续分布（例如单纯由树突滤波造成的上升时间—幅度连续谱）不应被强行切成几类。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from .features import FEATURES

LOG_FEATURES = {"amplitude", "rise_10_90", "decay_tau", "half_width", "charge", "max_rise_slope"}


@dataclass
class ClusterResult:
    labels: np.ndarray                  # 与输入事件等长；-1 表示噪声/未参与
    embedding: np.ndarray               # (n, 2)，未参与的事件为 NaN
    valid: np.ndarray                   # bool，参与聚类的事件
    k: int
    method: str
    representation: str
    silhouette: float
    bic: dict = field(default_factory=dict)       # k -> BIC
    delta_bic: float = np.nan                     # BIC(k=1) - BIC(best k>1)；>10 视为强证据
    multi_cluster_supported: bool = False
    summary: pd.DataFrame | None = None
    mean_waves: dict = field(default_factory=dict)
    mean_waves_norm: dict = field(default_factory=dict)
    explained_variance: np.ndarray | None = None
    message: str = ""


def _matrix(df: pd.DataFrame, waves: np.ndarray, representation: str, features: list[str],
            n_components: int):
    if representation == "waveform":
        pk = np.nanmax(waves, axis=1, keepdims=True)
        valid = np.isfinite(waves).all(axis=1) & (pk[:, 0] > 0)
        X = waves[valid] / pk[valid]
        # 轻度平滑抑制单个事件上的高频噪声，否则噪声会被 PCA 当成"形状差异"
        kw = max(X.shape[1] // 100, 1)
        if kw > 1:
            X = np.apply_along_axis(lambda r: np.convolve(r, np.ones(kw) / kw, mode="same"), 1, X)
        pca = PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]))
        Z = pca.fit_transform(X)
        Z = np.clip(Z / Z[:, 0].std(), -5, 5)
        return Z, valid, pca.explained_variance_ratio_
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    for j, f in enumerate(features):
        if f in LOG_FEATURES:
            with np.errstate(invalid="ignore", divide="ignore"):
                X[:, j] = np.log(X[:, j])
    valid = np.isfinite(X).all(axis=1)
    Z = RobustScaler().fit_transform(X[valid])
    # 截尾，避免个别离群事件主导
    Z = np.clip(Z, -5, 5)
    return Z, valid, None


def cluster_events(df: pd.DataFrame, waves: np.ndarray, representation: str = "waveform",
                   features: list[str] | None = None, method: str = "gmm", k: int | None = None,
                   k_max: int = 5, embedding: str = "pca", n_components: int = 3,
                   min_cluster_size: int = 20, seed: int = 0) -> ClusterResult:
    features = features or ["amplitude", "rise_10_90", "decay_tau", "half_width"]
    n = len(df)
    Z, valid, evr = _matrix(df, waves, representation, features, n_components)
    nv = len(Z)
    labels_full = np.full(n, -1, int)
    emb_full = np.full((n, 2), np.nan)
    if nv < 30:
        return ClusterResult(labels_full, emb_full, valid, 0, method, representation, np.nan,
                             message=f"有效事件只有 {nv} 个，不足以聚类（至少需要 30 个）。")

    # ---- 是否存在多簇：GMM 的 BIC 曲线 ----
    bic = {}
    gmms = {}
    for kk in range(1, min(k_max, nv // 10) + 1):
        g = GaussianMixture(kk, covariance_type="full", n_init=3, random_state=seed).fit(Z)
        bic[kk] = float(g.bic(Z))
        gmms[kk] = g
    best_multi = min((kk for kk in bic if kk > 1), key=lambda kk: bic[kk], default=1)
    delta_bic = bic[1] - bic[best_multi] if best_multi > 1 else np.nan

    if method == "gmm":
        # 自动选 k：BIC 最小，且不允许出现占比 <5% 的碎簇（多半是离群/误检事件）
        if k is None:
            ok = [q for q in bic if np.bincount(gmms[q].predict(Z), minlength=q).min() >= max(0.05 * nv, 10)]
            kk = min(ok, key=bic.get) if ok else 1
        else:
            kk = k
        if kk not in gmms:
            gmms[kk] = GaussianMixture(kk, covariance_type="full", n_init=3, random_state=seed).fit(Z)
        lab = gmms[kk].predict(Z)
    elif method == "kmeans":
        if k is None:
            sil = {}
            for kk in range(2, min(k_max, nv // 10) + 1):
                l_ = KMeans(kk, n_init=10, random_state=seed).fit_predict(Z)
                sil[kk] = silhouette_score(Z, l_)
            k = max(sil, key=sil.get) if sil else 1
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z) if k > 1 else np.zeros(nv, int)
    elif method == "hdbscan":
        lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Z)
    else:
        raise ValueError(method)

    # 按衰减 τ 从快到慢重新编号，保证不同细胞间簇编号含义一致
    uniq = [u for u in np.unique(lab) if u >= 0]
    tau = pd.to_numeric(df["decay_tau"], errors="coerce").to_numpy()[valid]
    order = sorted(uniq, key=lambda u: np.nanmedian(tau[lab == u]))
    remap = {u: i for i, u in enumerate(order)}
    lab = np.array([remap.get(u, -1) for u in lab])
    kfin = len(uniq)
    sil = float(silhouette_score(Z, lab)) if kfin > 1 and (lab >= 0).sum() > kfin else np.nan

    if embedding == "umap":
        try:
            import umap
            E = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(30, nv - 1)).fit_transform(Z)
        except Exception:
            E = PCA(2).fit_transform(Z) if Z.shape[1] > 2 else Z[:, :2]
    else:
        E = PCA(2).fit_transform(Z) if Z.shape[1] > 2 else Z[:, :2]

    labels_full[valid] = lab
    emb_full[valid] = E
    supported = bool(np.isfinite(delta_bic) and delta_bic > 10 and (np.isnan(sil) or sil > 0.25))
    if kfin <= 1:
        msg = "数据更支持单一连续分布（未发现可分离的亚群）。"
    elif supported:
        msg = f"多簇结构得到支持：ΔBIC = {delta_bic:.0f}（>10 为强证据），轮廓系数 = {sil:.2f}。"
    else:
        msg = (f"分成了 {kfin} 簇，但证据偏弱（ΔBIC = {delta_bic:.0f}，轮廓系数 = {sil:.2f}）；"
               "可能只是连续分布被切开，解读时请谨慎。")
    res = ClusterResult(labels_full, emb_full, valid, kfin, method, representation, sil, bic,
                        float(delta_bic) if np.isfinite(delta_bic) else np.nan, supported,
                        explained_variance=evr, message=msg)
    res.summary = cluster_summary(df, labels_full)
    for u in range(kfin):
        m = labels_full == u
        mw = np.nanmean(waves[m], axis=0)
        res.mean_waves[u] = mw
        res.mean_waves_norm[u] = mw / np.nanmax(mw)
    return res


def cluster_summary(df: pd.DataFrame, labels: np.ndarray, duration_s: float | None = None) -> pd.DataFrame:
    rows = []
    ntot = int((labels >= 0).sum())
    for u in sorted(set(labels[labels >= 0])):
        sub = df[labels == u]
        r = {"cluster": int(u) + 1, "n": len(sub), "fraction": len(sub) / ntot if ntot else np.nan}
        if duration_s:
            r["frequency"] = len(sub) / duration_s
        for f in FEATURES:
            r[f] = float(pd.to_numeric(sub[f], errors="coerce").median())
        rows.append(r)
    return pd.DataFrame(rows)
