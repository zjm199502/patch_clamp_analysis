"""分析会话：一个细胞（记录）一个 CellSession，多个细胞组成 Project。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import detection, features, preprocess
from ..io import Recording

DEFAULT_PARAMS = dict(
    direction="negative", window_ms=30.0, tau_rise_ms=0.4, tau_decay_ms=4.0,
    lowpass_hz=1000.0, highpass_hz=0.0, notch_hz=50.0, detrend_s=1.0,
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
    note: str = ""
    folders: list[str] = field(default_factory=list)
    detected_signature: str = ""
    failure: str = ""
    event_family: str = ""

    def family(self):
        if self.event_family:
            return self.event_family
        for name, preset in detection.PRESETS.items():
            if all(self.params[key] == value for key, value in preset.items()):
                return "IPSC" if "IPSC" in name else "EPSC"
        return ""

    @property
    def name(self) -> str:
        return self.rec.name

    def clear_qc_results(self):
        for key in ("detection_limit_pA", "fp_rate_hz", "fp_fraction"):
            self.qc.pop(key, None)

    def clear_detection(self, *, preprocessing=False):
        self.det = self.events = self.waves = self.t_wave = None
        self.clear_qc_results()
        if preprocessing:
            self.processed = None
            self.qc.clear()

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
                                                     p["threshold"], unit=self.rec.unit, packed=p.get("miniml_pipeline") == "整批滑窗（实验）")
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
        from .provenance import signature
        self.detected_signature = signature(self)
        self.failure = ""
        return len(df)

    def manual_candidate(self, left, right):
        if self.events is None or self.processed is None or self.t_wave is None:
            raise ValueError("请先运行事件检测。")
        from .provenance import signature
        if self.detected_signature and self.detected_signature != signature(self):
            raise ValueError("记录或参数已改变，请重新检测后再补录。")
        fs = self.rec.fs
        start, stop = sorted(round(value * fs) for value in (left, right))
        if start < 0 or stop >= len(self.processed) or stop - start < 3:
            raise ValueError("选区超出记录或太短。")
        win = len(self.t_wave) - int(np.count_nonzero(self.t_wave < 0))
        margin = 2 * win + int(.004 * fs) + 8
        offset = max(0, start - margin)
        df, waves, axis = features.extract_events(self.processed[offset:stop+margin+1], fs, [start-offset],
                self.params["direction"], win * 1000 / fs, peak_stop=stop-offset)
        if df.empty or not np.array_equal(axis, self.t_wave):
            raise ValueError("没有可测量的事件，或基线/衰减窗口不足。")
        for column in ("peak_idx", "onset_idx"):
            df[column] += offset
        peak = int(df.peak_idx.iloc[0])
        if peak in (start, stop) or (abs(self.events.peak_idx - peak) <= max(1, round(.001 * fs))).any():
            raise ValueError("峰在选区边界，或已有事件位于该峰附近。")
        df["time_s"] = df.onset_idx / fs
        df["manual"], df["excluded"], df["cluster"] = True, False, -1
        df["selection_start_s"], df["selection_end_s"] = start/fs, stop/fs
        return df, waves

    def accept_manual(self, candidate):
        df, waves = candidate
        previous = self.events.copy()
        if "manual" not in previous:
            previous["manual"] = False
        merged = pd.concat([previous, df], ignore_index=True) if len(previous) else df.copy()
        order = np.argsort(merged.time_s.to_numpy(), kind="stable")
        self.events = merged.iloc[order].reset_index(drop=True)
        self.events["iei"] = self.events.time_s.diff()
        self.events["inst_freq"] = 1 / self.events.iei
        self.events["cluster"] = -1
        self.waves = np.concatenate((self.waves, waves))[order]
        self.clear_qc_results()

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
    generation: str = ""

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

    def export_review_summary(self, path):
        rows = []
        for cell in self.cells:
            if cell.events is None:
                continue
            row = cell.summary()
            included = cell.mask()
            manual = cell.events.get("manual", pd.Series(False, index=cell.events.index)).fillna(False).to_numpy(bool)
            row.update(event_family=cell.family(), total_before_exclusion=len(included), excluded_events=int((~included).sum()),
                       manual_included=int((manual & included).sum()), note=cell.note)
            row.update({f"folder_{level}": name for level, name in enumerate(cell.folders, 1)})
            rows.append(row)
        if not rows:
            raise ValueError("尚无已检测的记录。")
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
