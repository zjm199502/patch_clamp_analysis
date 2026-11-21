"""Compare miniML CPU and GPU inference on the same ABF segment and threshold replay."""
import argparse
import os
from statistics import median
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("abf")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--direction", choices=["negative", "positive"], default="negative")
    parser.add_argument("--window-ms", type=float, default=30)
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument("--cpu-only", action="store_true")
    backends.add_argument("--gpu-python", "--cuda-python", help="Python executable with CUDA/MPS PyTorch, numpy and h5py")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.seconds <= 0 or args.window_ms <= 0:
        parser.error("Threads, duration and window length must be positive.")
    for key in ("TF_NUM_INTRAOP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import numpy as np
    import tensorflow as tf
    from patchclamp_gui import io
    from patchclamp_gui.core import gpu as accelerator
    from patchclamp_gui.core.detection import clear_model_cache
    from patchclamp_gui.core.session import CellSession

    if args.gpu_python:
        print("GPU device:", accelerator.probe(args.gpu_python))
    if not args.cpu_only and not args.gpu_python and not tf.config.list_physical_devices("GPU"):
        parser.error("TensorFlow cannot see a GPU. Use --cpu-only for the CPU baseline.")
    rec = io.load_abf(args.abf)
    rec.data = rec.data[:round(args.seconds * rec.fs)].copy()
    print(f"{rec.duration:g} s, {rec.fs:g} Hz, {rec.unit}", flush=True)
    results = []
    for device in (["CPU:0"] if args.cpu_only else ["CPU:0", "GPU:0"]):
        accelerator.configure(args.gpu_python if device == "GPU:0" and args.gpu_python else "")
        clear_model_cache()
        cell = CellSession(rec)
        cell.params.update(direction=args.direction, window_ms=args.window_ms, miniml_pipeline="整批滑窗（实验）")
        with tf.device("CPU:0" if args.gpu_python else device):
            cell.run_preprocess()
            cell.run_detection()
            cached = cell.det
            cell.params["threshold"] = .75
            full, replay = [], []
            for _ in range(3):
                for previous, times in ((None, full), (cached, replay)):
                    cell.det = previous
                    start = perf_counter()
                    cell.run_preprocess()
                    cell.run_detection()
                    times.append(perf_counter() - start)
                    if previous is None:
                        expected = cell.events
                        waves = cell.waves
                    elif not cell.events.equals(expected) or not np.array_equal(cell.waves, waves, equal_nan=True):
                        raise ValueError("Threshold replay differs from full detection.")
            print(f"{device}: {len(cell.events)} events, full {median(full):.3f} s, replay {median(replay):.3f} s")
            if args.gpu_python and device == "GPU:0" and cell.det.params["backend"] != accelerator.backend():
                raise RuntimeError(accelerator.failure or "Model used TensorFlow fallback")
            results.append((cell.events, cell.det.trace))
    accelerator.configure("")
    if len(results) == 2:
        (cpu, cpu_trace), (gpu, gpu_trace) = results
        peaks = set(gpu.peak_idx)
        weak = cpu[cpu.amplitude <= cpu.amplitude.quantile(.25)]
        for label, frame in (("all", cpu), ("smallest amplitudes", weak), ("overlapping", cpu[cpu.overlap.astype(bool)])):
            print(f"Identical peak samples ({label}): {len(peaks.intersection(frame.peak_idx))}/{len(frame)}")
        print("GPU-only peak samples:", len(peaks.difference(cpu.peak_idx)))
        print("Max prediction difference:", np.max(np.abs(cpu_trace - gpu_trace)))


if __name__ == "__main__":
    main()
