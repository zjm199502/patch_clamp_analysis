"""Standalone CUDA worker for the optional external PyTorch environment."""
import json
import os
from pathlib import Path
import sys

if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(None)
    dll_dirs = [p for p in (Path(sys.prefix) / "Library/bin",
                            Path(sys.prefix) / "Lib/site-packages/torch/lib") if p.is_dir()]
    os.environ["PATH"] = os.pathsep.join([*map(str, dll_dirs), os.environ.get("PATH", "")])
    dll_handles = [os.add_dll_directory(p) for p in dll_dirs]

import h5py
import numpy as np
import torch
import torch.nn.functional as F


class MiniModel:
    def __init__(self, path, graph_hash):
        import hashlib
        with h5py.File(path) as model:
            graph = model.attrs["model_config"]
            if isinstance(graph, str):
                graph = graph.encode()
            if hashlib.sha256(graph).hexdigest() != graph_hash:
                raise ValueError("此模型结构使用 TensorFlow 检测")
            weights = model["model_weights"]
            arrays = [torch.as_tensor(weights[layer][name][()], device="cuda")
                      for layer in weights.attrs["layer_names"]
                      for name in weights[layer].attrs["weight_names"]]
        values = iter(arrays)
        self.convs = []
        for _ in range(4):
            kernel, bias, gamma, beta, mean, var = [next(values) for _ in range(6)]
            scale = gamma / torch.sqrt(var + .001)
            self.convs.append(((kernel * scale).permute(2, 1, 0).contiguous(),
                               (bias - mean) * scale + beta))
        self.lstm = torch.nn.LSTM(80, 96, batch_first=True, bidirectional=True).to("cuda")
        for suffix in ("", "_reverse"):
            kernel, recurrent, bias = [next(values) for _ in range(3)]
            getattr(self.lstm, "weight_ih_l0" + suffix).copy_(kernel.T)
            getattr(self.lstm, "weight_hh_l0" + suffix).copy_(recurrent.T)
            getattr(self.lstm, "bias_ih_l0" + suffix).copy_(bias)
            getattr(self.lstm, "bias_hh_l0" + suffix).zero_()
        self.lstm.eval()
        self.linear, self.bias, self.output, self.intercept = values

    def predict(self, batch):
        low = batch.amin(1, keepdim=True)
        span = batch.amax(1, keepdim=True) - low
        x = ((batch - low) / torch.where(span == 0, 1, span)).unsqueeze(1)
        for i, (w, b) in enumerate(self.convs):
            x = F.leaky_relu(F.conv1d(x, w, b, padding=w.shape[-1] // 2), .3)
            if i < 3:
                x = F.avg_pool1d(x, [3, 2, 2][i])
        _, (hidden, _) = self.lstm(x.transpose(1, 2).contiguous())
        y = F.leaky_relu(hidden.sum(0) @ self.linear + self.bias, .3)
        return torch.sigmoid(y @ self.output + self.intercept).flatten()


def convolve_rows(x, weights):
    # NumPy "same" convention for an even kernel.
    k = weights.numel()
    padded = F.pad(x.reshape(-1, 1, x.shape[-1]), (k // 2, (k - 1) // 2))
    return F.conv1d(padded, weights.flip(0).view(1, 1, k)).reshape_as(x)


class Engine:
    model = model_key = None

    def run(self, op, args, arrays):
        x = torch.from_numpy(arrays["x"]).to("cuda")
        if op == "predict":
            if self.model_key != args["key"]:
                self.model = MiniModel(args["key"][0], args["graph_hash"])
                self.model_key = args["key"]
            return self.model.predict(x)
        if op == "peaks":
            # A plateau contributes its midpoint, rounded down, just like scipy.find_peaks.
            change = torch.where(x[1:] != x[:-1])[0] + 1
            left, right = change[:-1], change[1:]
            valid = (x[left] > x[left-1]) & (x[left] > x[right]) & (x[left] >= args["height"])
            return ((left[valid] + right[valid] - 1) // 2)
        if op == "smooth":
            return convolve_rows(x, torch.as_tensor(arrays["kernel"], device="cuda"))
        if op == "waveform":
            x = x / x.amax(1, keepdim=True)
            k = args["width"]
            if k > 1:
                x = convolve_rows(x, torch.full((k,), 1/k, dtype=x.dtype, device="cuda"))
            # Full PCA via the smaller symmetric Gram matrix; fix signs like sklearn.
            x -= x.mean(0)
            if x.shape[0] >= x.shape[1]:
                eig, axes = torch.linalg.eigh(x.T @ x)
                v = axes[:, -args["components"]:].flip([1])
                z = x @ v
            else:
                eig, axes = torch.linalg.eigh(x @ x.T)
                u = axes[:, -args["components"]:].flip([1])
                v = x.T @ u
                z = u * eig[-args["components"]:].flip([0]).clamp(min=0).sqrt()
            z *= torch.where(v[v.abs().argmax(0), torch.arange(v.shape[1], device="cuda")] < 0, -1, 1)
            evr = eig[-z.shape[1]:].flip([0]).clamp(min=0) / eig.sum()
            return torch.cat((z, evr[None]), 0)
        if op == "baseline":
            edges = np.linspace(0, len(x), args["segments"] + 1).astype(int)
            starts = torch.as_tensor(edges[:-1], device="cuda")
            med = torch.empty(len(starts), device="cuda", dtype=x.dtype)
            for width in np.unique(np.diff(edges)):
                rows = torch.as_tensor(np.flatnonzero(np.diff(edges) == width), device="cuda")
                seg = x[starts[rows, None] + torch.arange(int(width), device="cuda")]
                ordered = seg.sort(dim=1).values
                med[rows] = (ordered[:, (width-1)//2] + ordered[:, width//2]) / 2
                med[rows[seg.isnan().any(dim=1)]] = torch.nan
            return med
        nfft = args["nfft"]
        kernel = torch.fft.rfft(torch.as_tensor(arrays["kernel"], device="cuda"), nfft)
        spec = torch.fft.rfft(x, nfft)
        if op == "wiener":
            mag = kernel.abs().square()
            spec = spec * kernel.conj() / (mag + args["reg"] * mag.max())
            f = torch.fft.rfftfreq(nfft, 1/args["fs"], device="cuda", dtype=x.dtype)
            spec *= torch.exp(-.5 * (2*torch.pi*f*args["sigma"]).square())
        elif op == "reconstruct":
            spec *= kernel
        else:
            raise ValueError(op)
        return torch.fft.irfft(spec, nfft)[:len(x)]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.ones(1, device="cuda").cpu()
    if sys.argv[1] == "--probe":
        with torch.inference_mode():
            x = torch.ones((2, 4, 16), device="cuda")
            torch.nn.LSTM(16, 8, batch_first=True).cuda()(x)[0].cpu()
        print(torch.cuda.get_device_name())
        return
    directory = Path(sys.argv[1])
    engine = Engine()
    with torch.inference_mode():
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with np.load(directory / "input.npz", allow_pickle=False) as data:
                    result = engine.run(request["op"], request, data)
                np.save(directory / "result.npy", result.cpu().numpy(), allow_pickle=False)
                del result
                print(json.dumps({"ok": True}), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
