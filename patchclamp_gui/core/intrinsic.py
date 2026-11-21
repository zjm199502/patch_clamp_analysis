"""Analyze single-channel current-clamp recordings with a variable square step."""
import numpy as np
import pyabf
from . import gpu


def step_metrics(voltage, rate, start, stop, threshold):
    baseline = voltage[max(0, start - int(rate * .05)):start]
    segment = voltage[start:stop]
    if not baseline.size or len(segment) < 3 or not np.isfinite(np.r_[baseline, segment]).all():
        raise ValueError("阶跃或基线窗口无效")
    peaks = gpu.find_peaks(segment, height=threshold, prominence=15, distance=max(1, round(.002 * rate)))
    return dict(baseline_mV=float(np.median(baseline)), count=len(peaks),
                rate_Hz=len(peaks) * rate / len(segment), peak_ms=(peaks * 1000 / rate).tolist(),
                steady_mV=float(np.median(segment[-max(1, len(segment) // 10):])))


def analyze_abf(path, threshold=0.0):
    abf = pyabf.ABF(str(path))
    if abf.adcUnits != ["mV"] or abf.dacUnits[0] != "pA":
        raise ValueError("仅支持单通道 mV 电压记录、pA 方波电流指令。")
    epochs = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        command = abf.sweepC
        steps = {}
        for start, stop in zip(abf.sweepEpochs.p1s, abf.sweepEpochs.p2s):
            if start < .05 * abf.dataRate or stop - start < .1 * abf.dataRate or stop >= len(command):
                continue
            segment = command[start:stop]
            if np.all(segment == segment[0]) and np.all(command[start-int(.05*abf.dataRate):start] == command[start-1]):
                steps[start, stop] = float(segment[0] - command[start-1])
        epochs.append(steps)
    candidates = [key for key in set.intersection(*(set(steps) for steps in epochs))
                  if np.ptp([steps[key] for steps in epochs]) > 0]
    valid = []
    for start, stop in candidates:
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            if abf.sweepC[stop] != abf.sweepC[start-1]:
                break
        else:
            valid.append((start, stop))
    if len(valid) != 1:
        raise ValueError("无法唯一识别各 sweep 共有的变幅长方波，未生成结果。")
    start, stop = valid[0]
    rows = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        rows.append(dict(sweep=sweep+1, current_pA=epochs[sweep][start, stop],
                         **step_metrics(abf.sweepY, abf.dataRate, start, stop, threshold)))
    positive = [row["current_pA"] for row in rows if row["count"] and row["current_pA"] > 0]
    return dict(rows=rows, lowest_spiking_positive_pA=min(positive) if positive else None,
                start_s=start/abf.dataRate, duration_s=(stop-start)/abf.dataRate,
                threshold_mV=threshold, prominence_mV=15, min_peak_distance_ms=2)
