"""分析会话：一个细胞（记录）一个 CellSession，多个细胞组成 Project。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import detection, features, preprocess
from ..io import Recording

DEFAULT_PARAMS = dict(
    direction="negative", window_ms=30.0, tau_rise_ms=0.4, tau_decay_ms=4.0,
    lowpass_hz=2000.0, highpass_hz=0.0, notch_hz=0.0, detrend_s=1.0,
    detector="miniML", model="GC_lstm_model.h5", threshold=0.5,
    deconv_threshold_sd=4.0, deconv_smooth_ms=0.5, min_amplitude=0.0,
)


@dataclass
class CellSession:
    rec: Recording
    group: str = ""
    animal: str = ""
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    processed: np.ndarray | None = None
    det: detection.Detection | None = None
    events: pd.DataFrame | None = None
    waves: np.ndarray | None = None
    t_wave: np.ndarray | None = None
    qc: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rec.name

    def run_preprocess(self) -> np.ndarray:
        p = self.params
        self.processed = preprocess.preprocess(self.rec.data, self.rec.fs, p["lowpass_hz"],
                                               p["highpass_hz"], p["notch_hz"], p["detrend_s"])
        self.qc["noise_rms"] = preprocess.noise_rms(self.processed)
        return self.processed

    def detector_fn(self):
        """返回 y -> onsets 的函数（供质控模块用同样的参数重复检测）。"""
        p, fs = self.params, self.rec.fs
        if p["detector"] == "miniML":
            return lambda y: detection.detect_miniml(y, fs, p["direction"], p["window_ms"], p["model"],
                                                     p["threshold"], unit=self.rec.unit)
        return lambda y: detection.detect_deconv(y, fs, p["direction"], p["tau_rise_ms"], p["tau_decay_ms"],
                                                 p["deconv_threshold_sd"], p["deconv_smooth_ms"])

    def run_detection(self) -> int:
        if self.processed is None:
            self.run_preprocess()
        p = self.params
        self.det = self.detector_fn()(self.processed)
        df, w, tw = features.extract_events(self.processed, self.rec.fs, self.det.onsets, p["direction"],
                                            p["window_ms"], scores=self.det.scores,
                                            min_amplitude=p["min_amplitude"])
        df["excluded"] = False
        df["cluster"] = -1
        self.events, self.waves, self.t_wave = df, w, tw
        return len(df)

    # ---- 有效事件（剔除人工排除的） ----
    def mask(self) -> np.ndarray:
        if self.events is None:
            return np.zeros(0, bool)
        return ~self.events["excluded"].to_numpy(bool)

    def valid(self) -> tuple[pd.DataFrame, np.ndarray]:
        m = self.mask()
        df = self.events[m].reset_index(drop=True).copy()
        df["iei"] = df["time_s"].diff()
        return df, self.waves[m]

    def summary(self) -> dict:
        out = {"cell": self.name, "group": self.group, "animal": self.animal,
               "noise_rms": self.qc.get("noise_rms", np.nan)}
        if self.events is not None:
            df, w = self.valid()
            out.update(features.summarize(df, self.rec.duration, w, self.rec.fs))
            for k in ("detection_limit_pA", "fp_rate_hz", "fp_fraction"):
                if k in self.qc:
                    out[k] = self.qc[k]
            cl = df["cluster"].to_numpy()
            if (cl >= 0).any():
                for u in sorted(set(cl[cl >= 0])):
                    out[f"cluster{u + 1}_fraction"] = float(np.mean(cl[cl >= 0] == u))
                    out[f"cluster{u + 1}_frequency"] = float(np.sum(cl == u) / self.rec.duration)
        return out


@dataclass
class Project:
    cells: list[CellSession] = field(default_factory=list)

    def analyzed(self) -> list[CellSession]:
        return [c for c in self.cells if c.events is not None and len(c.events)]

    def cell_table(self) -> pd.DataFrame:
        return pd.DataFrame([c.summary() for c in self.cells])

    def events_table(self) -> pd.DataFrame:
        parts = []
        for c in self.analyzed():
            df, _ = c.valid()
            df.insert(0, "cell", c.name)
            df.insert(1, "group", c.group)
            df.insert(2, "animal", c.animal)
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def pooled_waves(self) -> tuple[pd.DataFrame, np.ndarray] | None:
        """合并所有细胞的事件与波形（要求采样率与窗口一致）。"""
        cs = self.analyzed()
        if not cs:
            return None
        L = {c.waves.shape[1] for c in cs}
        fs = {c.rec.fs for c in cs}
        if len(L) > 1 or len(fs) > 1:
            return None
        return self.events_table(), np.concatenate([c.valid()[1] for c in cs])

    def export(self, stem: str) -> list[str]:
        paths = []
        ev, ct = self.events_table(), self.cell_table()
        ev.to_csv(f"{stem}_events.csv", index=False, encoding="utf-8-sig")
        ct.to_csv(f"{stem}_cells.csv", index=False, encoding="utf-8-sig")
        paths += [f"{stem}_events.csv", f"{stem}_cells.csv"]
        return paths
