import pennylane as qml
import torch
import torch.nn as nn


class QuantumResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_qubits=4,
        n_qlayers=1,
        n_esteps=1,
        data_reupload=False,
        backend="default.qubit",
    ):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if n_qlayers <= 0:
            raise ValueError("n_qlayers must be positive")
        if n_esteps < 0:
            raise ValueError("n_esteps must be non-negative")

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_esteps = n_esteps
        self.data_reupload = data_reupload

        self.entry = nn.Linear(in_dim, n_qubits)
        self.exit = nn.Linear(n_qubits, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        wires = list(range(n_qubits))
        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            feats = inputs.transpose(1, 0)
            ry_params = [torch.arctan(feats[i]) for i in range(self.n_qubits)]
            rz_params = [torch.arctan(feats[i] ** 2) for i in range(self.n_qubits)]

            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(ry_params[i], wires=i)
                qml.RZ(rz_params[i], wires=i)

            for layer in range(self.n_qlayers):
                self._ansatz(weights[layer], wires)
                if self.data_reupload and layer < self.n_qlayers - 1:
                    for i in range(self.n_qubits):
                        qml.RY(ry_params[i], wires=i)
                        qml.RZ(rz_params[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in wires]

        self.vqc = qml.qnn.TorchLayer(circuit, {"weights": (n_qlayers, 3, n_qubits)})

    def _ansatz(self, params, wires):
        if self.n_qubits > 1 and self.n_esteps > 0:
            for k in range(self.n_esteps):
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[wires[i], wires[(i + k + 1) % self.n_qubits]])
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires[i])
            qml.RY(params[1][i], wires=wires[i])
            qml.RZ(params[2][i], wires=wires[i])

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        z = self.entry(flat)
        q = self.vqc(z)
        y = self.exit(z + q)
        y = self.norm(y + self.skip(flat))
        return y.reshape(*shape[:-1], -1)


def _base_init_kwargs(base_model_name, configs):
    if base_model_name in {"PAttn", "PatchTST"}:
        return {
            "patch_len": getattr(configs, "patch_len", 16),
            "stride": getattr(configs, "patch_stride", 8),
        }
    if base_model_name == "Pyraformer":
        return {
            "window_size": getattr(configs, "pyraformer_window_size", [4, 4]),
            "inner_size": getattr(configs, "pyraformer_inner_size", 3),
        }
    return {}


class QuantumHybridForecastModel(nn.Module):
    def __init__(self, configs, base_model_cls, base_model_name, *base_args, **base_kwargs):
        super().__init__()
        init_kwargs = _base_init_kwargs(base_model_name, configs)
        init_kwargs.update(base_kwargs)

        self.base_model = base_model_cls(configs, *base_args, **init_kwargs).float()
        self.output_dim = self._infer_output_dim(configs)
        self.qblock = QuantumResidualBlock(
            self.output_dim,
            self.output_dim,
            n_qubits=getattr(configs, "n_qubits", 4),
            n_qlayers=getattr(configs, "n_qlayers", 1),
            n_esteps=getattr(configs, "n_esteps", 1),
            data_reupload=getattr(configs, "data_reupload", False),
            backend=getattr(configs, "quantum_backend", "default.qubit"),
        )

    def _infer_output_dim(self, configs):
        seq_len = getattr(configs, "seq_len")
        label_len = getattr(configs, "label_len", seq_len)
        pred_len = getattr(configs, "pred_len", 1)
        enc_in = getattr(configs, "enc_in")
        dec_in = getattr(configs, "dec_in", enc_in)

        x_enc = torch.zeros(1, seq_len, enc_in, dtype=torch.float32)
        x_mark_enc = torch.zeros(1, seq_len, 4, dtype=torch.float32)
        x_dec = torch.zeros(1, label_len + pred_len, dec_in, dtype=torch.float32)
        x_mark_dec = torch.zeros(1, label_len + pred_len, 4, dtype=torch.float32)

        was_training = self.base_model.training
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if was_training:
            self.base_model.train()

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(f"Expected tensor forecast output, got {type(outputs)!r}")
        return int(outputs.size(-1))

    def _apply_qblock(self, outputs):
        if isinstance(outputs, tuple):
            head = outputs[0]
            if isinstance(head, torch.Tensor) and head.dim() == 3 and head.size(-1) == self.output_dim:
                head = self.qblock(head)
            return (head, *outputs[1:])
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 3 and outputs.size(-1) == self.output_dim:
            return self.qblock(outputs)
        return outputs

    def forward(self, *args, **kwargs):
        return self._apply_qblock(self.base_model(*args, **kwargs))


def build_quantum_model_class(base_model_cls, base_model_name):
    class Model(QuantumHybridForecastModel):
        def __init__(self, configs, *args, **kwargs):
            super().__init__(configs, base_model_cls, base_model_name, *args, **kwargs)

    Model.__name__ = "Model"
    Model.__qualname__ = "Model"
    return Model
