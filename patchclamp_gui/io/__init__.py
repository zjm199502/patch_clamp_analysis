"""数据读取：Axon ABF（pyabf）、HDF5，以及用于演示/测试的模拟数据。"""
from __future__ import annotations

import csv
import os
from bisect import bisect_left
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    data: np.ndarray            # 1D，已按 sweep 拼接
    fs: float                   # 采样率 Hz
    unit: str = "pA"
    name: str = ""
    path: str = ""
    channel: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fs

    def sample_range(self, start: float, end: float) -> tuple[int, int]:
        """Return samples whose trace times fall in [start, end)."""
        indices = range(len(self.data))
        def time_at(i):
            return i / self.fs
        first = bisect_left(indices, start, key=time_at)
        stop = bisect_left(indices, end, key=time_at)
        if first >= stop:
            raise ValueError("所选时间段没有采样点，请调整起点或终点。")
        return first, stop

    def export_samples(self, path, first: int, stop: int):
        if not 0 <= first < stop <= len(self.data):
            raise ValueError("采样范围超出记录。")
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", dir=Path(path).parent, delete=False,
                                    encoding="utf-8-sig", newline="") as stream:
                temporary = Path(stream.name)
                csv.writer(stream).writerow(["sample_index", "time_s", "relative_time_s",
                                             f"signal_{self.unit}" if self.unit else "signal"])
                for begin in range(first, stop, 65536):
                    end = min(begin + 65536, stop)
                    indices = np.arange(begin, end)
                    values = np.column_stack((indices, indices / self.fs,
                                              (indices - first) / self.fs, self.data[begin:end]))
                    np.savetxt(stream, values, delimiter=",", fmt=["%d", "%.17g", "%.17g", "%.17g"])
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def abf_info(path: str) -> dict:
    import pyabf

    abf = pyabf.ABF(path, loadData=False)
    return {
        "channels": list(abf.channelList),
        "channel_names": list(abf.adcNames),
        "units": list(abf.adcUnits),
        "sweeps": abf.sweepCount,
        "fs": float(abf.sampleRate),
        "protocol": abf.protocol,
        "sweep_length_s": abf.sweepPointCount / abf.sampleRate,
    }


def load_abf(path: str, channel: int = 0, sweeps: list[int] | None = None) -> Recording:
    """读取 ABF。gap-free 记录直接返回；episodic 记录把所选 sweep 首尾拼接。"""
    import pyabf

    abf = pyabf.ABF(path)
    if channel not in abf.channelList:
        raise IndexError(f"通道 {channel} 不存在，可用通道：{abf.channelList}")
    sweeps = list(abf.sweepList) if sweeps is None else sweeps
    parts = []
    for s in sweeps:
        abf.setSweep(sweepNumber=s, channel=channel)
        parts.append(np.asarray(abf.sweepY, dtype=float))
    data = np.concatenate(parts)
    meta = {
        "sweeps": len(sweeps),
        "sweep_points": int(abf.sweepPointCount),
        "protocol": abf.protocol,
        "abf_version": abf.abfVersionString,
    }
    return Recording(data=data, fs=float(abf.sampleRate), unit=abf.adcUnits[channel],
                     name=Path(path).name, path=str(path), channel=channel, meta=meta)


def load_h5(path: str, dataset: str = "mini_data", fs: float = 50_000.0,
            scaling: float = 1.0, unit: str = "pA") -> Recording:
    import h5py

    with h5py.File(path, "r") as f:
        data = np.asarray(f[dataset], dtype=float).ravel() * scaling
    return Recording(data=data, fs=fs, unit=unit, name=Path(path).name, path=str(path))


def load_any(path: str, channel: int = 0) -> Recording:
    suf = Path(path).suffix.lower()
    if suf == ".abf":
        return load_abf(path, channel=channel)
    if suf in (".h5", ".hdf5"):
        return load_h5(path)
    if suf in (".npy",):
        return Recording(data=np.load(path).astype(float).ravel(), fs=20_000.0,
                         name=Path(path).name, path=str(path))
    raise ValueError(f"不支持的文件类型：{suf}")


def biexp_kernel(fs: float, tau_rise_ms: float, tau_decay_ms: float,
                 length_ms: float | None = None) -> np.ndarray:
    """峰值归一化为 1 的双指数波形。"""
    tr, td = tau_rise_ms / 1e3, tau_decay_ms / 1e3
    if length_ms is None:
        length_ms = 10 * tau_decay_ms
    t = np.arange(int(round(length_ms / 1e3 * fs))) / fs
    k = np.exp(-t / td) - np.exp(-t / tr)
    return k / k.max()


def simulate_recording(duration_s: float = 60.0, fs: float = 20_000.0, rate_hz: float = 3.0,
                       direction: str = "negative", noise_sd: float = 2.5,
                       populations: list[dict] | None = None, seed: int = 0,
                       name: str = "simulated") -> Recording:
    """生成含多个动力学亚群的模拟微小突触电流记录（泊松发放 + 有色噪声）。

    populations 每项：dict(frac, amp, amp_cv, tau_rise, tau_decay)（ms / pA）。
    真值保存在 meta['truth'] 中。
    """
    rng = np.random.default_rng(seed)
    if populations is None:
        populations = [
            dict(frac=0.6, amp=18.0, amp_cv=0.35, tau_rise=0.3, tau_decay=3.0),
            dict(frac=0.4, amp=9.0, amp_cv=0.30, tau_rise=1.2, tau_decay=9.0),
        ]
    n = int(duration_s * fs)
    y = np.zeros(n)
    n_ev = rng.poisson(rate_hz * duration_s)
    onsets = np.sort(rng.uniform(0.05, duration_s - 0.1, n_ev))
    fr = np.array([p["frac"] for p in populations], float)
    pop = rng.choice(len(populations), size=n_ev, p=fr / fr.sum())
    amps = np.zeros(n_ev)
    for i, (t0, pi) in enumerate(zip(onsets, pop)):
        p = populations[pi]
        sig = np.sqrt(np.log(1 + p["amp_cv"] ** 2))
        a = p["amp"] * rng.lognormal(-sig ** 2 / 2, sig)
        jit = rng.normal(1, 0.1, 2).clip(0.6, 1.5)
        k = biexp_kernel(fs, p["tau_rise"] * jit[0], p["tau_decay"] * jit[1])
        i0 = int(t0 * fs)
        m = min(len(k), n - i0)
        y[i0:i0 + m] += a * k[:m]
        amps[i] = a
    # 有色噪声：白噪声经 2 kHz 低通
    from scipy.signal import butter, sosfiltfilt
    noise = rng.normal(0, 1, n)
    noise = sosfiltfilt(butter(2, 2000 / (fs / 2), output="sos"), noise)
    noise *= noise_sd / noise.std()
    drift = 3 * np.sin(2 * np.pi * np.arange(n) / fs / 20.0)
    sign = -1.0 if direction == "negative" else 1.0
    data = sign * y + noise + drift
    truth = {"onsets_s": onsets, "population": pop, "amplitude": amps}
    return Recording(data=data, fs=fs, unit="pA", name=name, meta={"truth": truth, "simulated": True})
