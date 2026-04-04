import pennylane as qml
import torch
import torch.nn as nn


class QuantumSSM(nn.Module):
    """
    Quantum State Space Model — 4-qubit selective recurrence.

    Uses the same VQC as cQLSTM (identical encoding + ansatz) but replaces
    the 4-gate LSTM cell with a 2-component SSM selective scan:

        combined_t  =  concat(x_t, h_t)                       [B, input+hidden]
        y_t         =  entry(combined_t)                       [B, n_qubits]
        qubits_t    =  VQC(y_t)                                [B, n_qubits]
        out_t       =  exit(y_t + qubits_t)                    [B, hidden*2]  ← residual

        g_raw, delta_raw  =  out_t.chunk(2)
        g_t     =  clamp( sigmoid(g_raw),  g_min, g_max )      [B, hidden]
        delta_t =  tanh( delta_raw )                           [B, hidden]

        h_t  =  (1 - g_t) * h_{t-1}  +  g_t * delta_t

    VQC (identical to cQLSTM)
    -------------------------
    Encoding per qubit i:
        H → RY(arctan(y_i)) → RZ(arctan(y_i²))
    Ansatz (repeated n_qlayers times):
        CNOT ring (n_esteps offsets) → RX/RY/RZ variational rotations

    Differences from cQLSTM
    -----------------------
    | Aspect            | cQLSTM              | QuantumSSM            |
    |-------------------|---------------------|-----------------------|
    | Cell state        | explicit c_t (LSTM) | none (SSM)            |
    | Gate decomp.      | f, i, g, o (×4)     | g, delta (×2)        |
    | exit projection   | n_qubits → hidden×4 | n_qubits → hidden×2   |
    | Gate scope        | per-dim (hidden)    | per-dim (hidden)      |
    | Gate dependency   | input + hidden      | input + hidden        |
    | Decay             | explicit decay_rate | implicit via (1-g_t)  |

    Interface
    ---------
    forward(x, hidden=None)
        x       : [B, T, input_size]
        returns : (outputs, h_t)
                  outputs : [B, T, hidden_size]
                  h_t     : [B, hidden_size]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_qubits: int = 4,
        n_qlayers: int = 1,
        n_esteps: int = 1,
        g_min: float = 0.05,
        g_max: float = 0.95,
        data_reupload: bool = False,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_qubits     = n_qubits
        self.n_qlayers    = n_qlayers
        self.n_vrotations = 3
        self.n_esteps     = n_esteps
        self.g_min        = g_min
        self.g_max        = g_max
        self.data_reupload = data_reupload

        # ── feature weighting (same as cQLSTM) ───────────────────────────
        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))

        # ── entry / exit (SSM uses hidden*2 instead of hidden*4) ─────────
        self.entry = nn.Linear(input_size + hidden_size, n_qubits)
        self.exit  = nn.Linear(n_qubits, hidden_size * 2)

        # ── VQC with optional data re-uploading ──────────────────────────
        self.wires  = list(range(n_qubits))
        self._dev   = qml.device(backend, wires=n_qubits)

        _data_reupload = data_reupload

        @qml.qnode(self._dev, interface="torch")
        def _qnode(inputs, weights):
            features    = inputs.transpose(1, 0)
            ry_params   = [torch.arctan(f)    for f in features]
            rz_params   = [torch.arctan(f**2) for f in features]

            # Initial encoding
            for i in range(n_qubits):
                qml.Hadamard(wires=self.wires[i])
                qml.RY(ry_params[i], wires=self.wires[i])
                qml.RZ(rz_params[i], wires=self.wires[i])

            # Variational layers with optional data re-uploading
            for layer in range(n_qlayers):
                self._ansatz(weights[layer], self.wires)
                if _data_reupload and layer < n_qlayers - 1:
                    for i in range(n_qubits):
                        qml.RY(ry_params[i], wires=self.wires[i])
                        qml.RZ(rz_params[i], wires=self.wires[i])

            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]

        weight_shapes = {"weights": (n_qlayers, self.n_vrotations, n_qubits)}
        self.VQC = qml.qnn.TorchLayer(_qnode, weight_shapes)

    def _ansatz(self, params, wires_type):
        # Entangling layer
        for k in range(self.n_esteps):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % self.n_qubits]])
        # Variational layer
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Parameters
        ----------
        x      : [B, T, input_size]
        hidden : optional initial h_t  [B, hidden_size]

        Returns
        -------
        outputs : [B, T, hidden_size]
        h_t     : [B, hidden_size]  — final hidden state
        """
        B, T, _ = x.shape

        h_t = (
            torch.zeros(B, self.hidden_size, device=x.device)
            if hidden is None
            else hidden
        )

        # Learnable per-feature scaling (same as cQLSTM)
        x = x * self.feature_weighting.exp()

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, h_t), dim=1)       # [B, input+hidden]
            y        = self.entry(combined)                 # [B, n_qubits]
            qubits   = self.VQC(y)                         # [B, n_qubits]
            out      = self.exit(y + qubits)               # [B, hidden*2]  residual

            g_raw, delta_raw = out.chunk(2, dim=1)

            g_t     = torch.clamp(torch.sigmoid(g_raw), self.g_min, self.g_max)
            delta_t = torch.tanh(delta_raw)

            h_t = (1.0 - g_t) * h_t + g_t * delta_t      # [B, hidden]
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t            # [B, T, hidden], [B, hidden]
