"""Offline Windows miniML inference with the DirectML execution provider."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


BATCH = 128


def model_bytes(path, graph_hash):
    with h5py.File(path) as model:
        config = model.attrs["model_config"]
        if isinstance(config, str):
            config = config.encode()
        if hashlib.sha256(config).hexdigest() != graph_hash:
            raise ValueError("miniML model changed while loading")
        weights = model["model_weights"]
        values = iter([weights[layer][name][()] for layer in weights.attrs["layer_names"]
                       for name in weights[layer].attrs["weight_names"]])

    nodes, initializers = [], []

    def constant(name, array):
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    x = "windows"
    for i in range(4):
        kernel, bias, gamma, beta, mean, variance = [next(values) for _ in range(6)]
        scale = gamma / np.sqrt(variance + .001)
        w = constant(f"w{i}", (kernel * scale).transpose(2, 1, 0))
        b = constant(f"b{i}", (bias - mean) * scale + beta)
        pad = kernel.shape[0] // 2
        x = node("Conv", [x, w, b], f"conv{i}", pads=[pad, pad])
        x = node("LeakyRelu", [x], f"activation{i}", alpha=.3)
        if i < 3:
            stride = (3, 2, 2)[i]
            x = node("AveragePool", [x], f"pool{i}", kernel_shape=[stride], strides=[stride])

    x = node("Transpose", [x], "sequence", perm=[2, 0, 1])
    kernels, recurrent, biases = [], [], []
    for _ in range(2):
        # Keras stores input/forget/cell/output. ONNX uses input/output/forget/cell.
        kernel, state, bias = [next(values) for _ in range(3)]
        def reorder(a):
            gates = np.split(a, 4, axis=0)
            return np.concatenate([gates[j] for j in (0, 3, 1, 2)])
        kernels.append(reorder(kernel.T))
        recurrent.append(reorder(state.T))
        biases.append(np.concatenate((reorder(bias), np.zeros_like(bias))))
    nodes.append(helper.make_node("LSTM", [x, constant("W", kernels), constant("R", recurrent),
                                            constant("B", biases)], ["sequence_out", "hidden"],
                                  direction="bidirectional", hidden_size=96))
    x = node("ReduceSum", ["hidden", constant("axis", np.array([0], dtype=np.int64))], "merged", keepdims=0)
    dense, bias, output, intercept = values
    x = node("Gemm", [x, constant("dense", dense), constant("bias", bias)], "dense_out")
    x = node("LeakyRelu", [x], "dense_act", alpha=.3)
    x = node("Gemm", [x, constant("output", output), constant("intercept", intercept)], "logits")
    node("Sigmoid", [x], "scores")
    graph = helper.make_graph(nodes, "miniML",
                              [helper.make_tensor_value_info("windows", TensorProto.FLOAT, [BATCH, 1, 600])],
                              [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [BATCH, 1])],
                              initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8).SerializeToString()


def session(model):
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("DirectML is unavailable")
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    result = ort.InferenceSession(model, sess_options=options, providers=["DmlExecutionProvider"])
    if "DmlExecutionProvider" not in result.get_providers():
        raise RuntimeError("DirectML initialization failed")
    return result


def predict(engine, batch):
    output = []
    windows = np.empty((BATCH, 1, 600), dtype=np.float32)
    for start in range(0, len(batch), BATCH):
        x = batch[start:start + BATCH]
        v = x - x.min(axis=1, keepdims=True)
        span = v.max(axis=1, keepdims=True)
        windows[:len(x), 0] = v / np.where(span == 0, 1, span)
        windows[len(x):] = 0
        output.append(engine.run(None, {"windows": windows})[0][:len(x), 0])
    return np.concatenate(output)


def main():
    if sys.argv[1] == "--runtime-info":
        print(json.dumps(ort.get_available_providers()))
        return
    if sys.argv[1] == "--probe":
        v = helper.make_tensor_value_info("x", TensorProto.FLOAT, [16, 16])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [16, 16])
        graph = helper.make_graph([helper.make_node("MatMul", ["x", "x"], ["y"])], "probe", [v], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        result = session(model.SerializeToString()).run(None, {"x": np.eye(16, dtype=np.float32)})[0]
        np.testing.assert_array_equal(result, np.eye(16))
        print("Windows GPU (DirectML)")
        return
    directory = Path(sys.argv[1])
    model_key = engine = None
    for line in sys.stdin:
        try:
            args = json.loads(line)
            if args["op"] != "predict":
                raise ValueError("Unsupported DirectML operation")
            if args["key"] != model_key:
                engine = session(model_bytes(args["key"][0], args["graph_hash"]))
                model_key = args["key"]
            with np.load(directory / "input.npz", allow_pickle=False) as data:
                scores = predict(engine, data["x"])
            np.save(directory / "result.npy", scores, allow_pickle=False)
            print(json.dumps({"ok": True}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
