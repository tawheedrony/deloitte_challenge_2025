import pennylane as qml
import torch
import torch.nn as nn


class cQSSM(nn.Module):
    """
    cQSSM — Improved Quantum State Space Model.

    Builds on QSSM with four targeted improvements:
      1. Explicit cell state  (slow + fast memory, like LSTM's c_t / h_t split)
      2. Decay on retention   (proven +0.7 pp from cLSTM vs LSTM)
      3. LayerNorm on output  (stabilises early training; QSSM had ep1 val/R²=−9)
      4. Separate input gate  (controls how much candidate updates the cell)

    Recurrence
    ----------
        combined_t  = concat(x_t, h_t)                         [B, input+hidden]
        y_t         = entry(combined_t)                         [B, n_qubits]
        qubits_t    = VQC(y_t)                                  [B, n_qubits]
        out_t       = exit(y_t + qubits_t)                      [B, hidden*3]  residual

        g_raw, i_raw, delta_raw  = out_t.chunk(3)
        g_t     = clamp( sigmoid(g_raw), g_min, g_max )         retention gate
        i_t     = sigmoid(i_raw)                                 input gate
        delta_t = tanh(delta_raw)                                candidate

        c_t = (1 - g_t) * (1 - decay_rate) * c_{t-1}
              + i_t * delta_t                                    cell state (slow)
        h_t = LayerNorm( g_t * tanh(c_t) )                      hidden (fast)

    Differences from QSSM
    ---------------------
    | Aspect          | QSSM               | cQSSM                    |
    |-----------------|--------------------|--------------------------|
    | Cell state      | none (single h_t)  | explicit c_t             |
    | Input gate      | implicit via g     | explicit i_t             |
    | Decay           | none               | (1 - decay_rate) on c_t  |
    | Output norm     | none               | LayerNorm                |
    | exit projection | n_qubits → h*2     | n_qubits → h*3           |

    Interface
    ---------
    forward(x, hidden=None)
        x      : [B, T, input_size]
        hidden : optional (h_0, c_0) tuple
        returns: (outputs [B, T, hidden], h_T [B, hidden])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_qubits: int = 4,
        n_qlayers: int = 2,
        n_esteps: int = 1,
        decay_rate: float = 0.1,
        g_min: float = 0.05,
        g_max: float = 0.95,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_qubits    = n_qubits
        self.n_qlayers   = n_qlayers
        self.n_esteps    = n_esteps
        self.decay_rate  = decay_rate
        self.g_min       = g_min
        self.g_max       = g_max
        self.n_vrotations = 3

        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))

        # entry bottleneck: concat → n_qubits
        self.entry = nn.Linear(input_size + hidden_size, n_qubits)
        # exit: n_qubits → hidden*3  (g, i, delta)
        self.exit  = nn.Linear(n_qubits, hidden_size * 3)

        self.norm = nn.LayerNorm(hidden_size)

        # VQC — identical circuit to cQLSTM / QSSM
        self.wires = list(range(n_qubits))
        self._dev  = qml.device(backend, wires=n_qubits)

        @qml.qnode(self._dev, interface="torch")
        def _qnode(inputs, weights):
            features  = inputs.transpose(1, 0)
            ry_params = [torch.arctan(f)    for f in features]
            rz_params = [torch.arctan(f**2) for f in features]
            for i in range(n_qubits):
                qml.Hadamard(wires=self.wires[i])
                qml.RY(ry_params[i], wires=self.wires[i])
                qml.RZ(rz_params[i], wires=self.wires[i])
            qml.layer(self._ansatz, n_qlayers, weights, wires_type=self.wires)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]

        weight_shapes = {"weights": (n_qlayers, self.n_vrotations, n_qubits)}
        self.VQC = qml.qnn.TorchLayer(_qnode, weight_shapes)

    def _ansatz(self, params, wires_type):
        for k in range(self.n_esteps):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % self.n_qubits]])
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def forward(self, x: torch.Tensor, hidden=None):
        B, T, _ = x.shape

        if hidden is None:
            h_t = torch.zeros(B, self.hidden_size, device=x.device)
            c_t = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            h_t, c_t = hidden

        x = x * self.feature_weighting.exp()

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, h_t), dim=1)        # [B, input+hidden]
            y        = self.entry(combined)                  # [B, n_qubits]
            qubits   = self.VQC(y)                           # [B, n_qubits]
            out      = self.exit(y + qubits)                 # [B, hidden*3]

            g_raw, i_raw, delta_raw = out.chunk(3, dim=1)

            g_t     = torch.clamp(torch.sigmoid(g_raw), self.g_min, self.g_max)
            i_t     = torch.sigmoid(i_raw)
            delta_t = torch.tanh(delta_raw)

            # cell: decay-attenuated retention + gated candidate
            c_t = (1.0 - g_t) * (1.0 - self.decay_rate) * c_t + i_t * delta_t
            # hidden: gated read of cell, normalised
            h_t = self.norm(g_t * torch.tanh(c_t))

            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t              # [B, T, hidden], [B, hidden]
