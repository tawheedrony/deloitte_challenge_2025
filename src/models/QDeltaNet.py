import sys
from pathlib import Path

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

_LIBS = Path(__file__).parent.parent.parent / "libs" / "deltanet"
sys.path.insert(0, str(_LIBS))
from deltanet import chunk_batched_delta_rule_forward  # noqa: E402


def _delta_step(q, k, v, beta, S):
    """
    Batched recurrent delta-rule step.

    q, k, v : [B, d]
    beta    : [B, 1]
    S       : [B, d, d]
    returns : h [B, d],  S_new [B, d, d]
    """
    v_old = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)
    v_new = beta * v + (1.0 - beta) * v_old
    S = (S
         - torch.bmm(v_old.unsqueeze(-1), k.unsqueeze(1))
         + torch.bmm(v_new.unsqueeze(-1), k.unsqueeze(1)))
    h = torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)
    return h, S


class QDeltaNet(nn.Module):
    """
    Quantum DeltaNet — 3 independent VQCs with recurrent context.

    Each of Q, K, V gets its own VQC, and every VQC is conditioned on
    concat(x_t, h_{t-1}) so it sees accumulated sequence history.
    The delta-rule memory matrix S provides O(d²) capacity recurrence,
    while h_t (the per-step readout) closes the feedback loop to the VQCs.

    Architecture  (default, shared_vqc=False)
    -----------------------------------------
    For role ∈ {Q, K, V}:
        combined  =  concat( x_t * exp(fw),  h_{t-1} )  [B, input+hidden]
        y_role    =  entry_role(combined)                 [B, n_qubits]
        role_raw  =  exit_role( y_role + VQC_role(y_role) )  [B, hidden]

    q_t = normalize(Q_raw),  k_t = normalize(K_raw)
    v_t = V_raw / alpha
    beta_t = sigmoid( beta_proj(x_t) )

    h_t, S  =  delta_step(q_t, k_t, v_t, beta_t, S_{t-1})

    outputs  =  LayerNorm( stack(h_t) )    [B, T, hidden]

    Legacy mode (shared_vqc=True)
    ------------------------------
    Loads old checkpoints that used a single shared VQC on raw input
    (no recurrence).  Key: model.entry.weight present in state_dict.

    Parameters
    ----------
    input_size  : int
    hidden_size : int   — d; Q/K/V and output dimension
    n_qubits    : int   — VQC bottleneck per circuit
    n_qlayers   : int   — variational ansatz repetitions
    n_esteps    : int   — CNOT ring offsets per ansatz layer
    neg_eigen   : bool  — alpha=2 allows negative eigenvalues
    shared_vqc  : bool  — legacy compat flag (auto-set by eval.py)
    chunk_size  : int   — unused (kept for API compat)
    backend     : str   — PennyLane device
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_qubits: int = 4,
        n_qlayers: int = 1,
        n_esteps: int = 1,
        neg_eigen: bool = False,
        shared_vqc: bool = False,
        chunk_size: int = 0,        # API compat only
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_qubits     = n_qubits
        self.n_qlayers    = n_qlayers
        self.n_vrotations = 3
        self.n_esteps     = n_esteps
        self.alpha        = 2 if neg_eigen else 1
        self.shared_vqc   = shared_vqc
        del chunk_size  # API compat only — recurrent mode processes step-by-step

        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))

        if shared_vqc:
            # ── legacy: shared VQC on raw input, no recurrence ────────────
            self.entry  = nn.Linear(input_size, n_qubits)
            self.exit_q = nn.Linear(n_qubits, hidden_size)
            self.exit_k = nn.Linear(n_qubits, hidden_size)
            self.exit_v = nn.Linear(n_qubits, hidden_size)
            self.beta_proj = nn.Linear(input_size, 1)
            self.proj_out  = nn.Linear(hidden_size, hidden_size)
            self.norm      = nn.LayerNorm(hidden_size)
        else:
            # ── default: 3 VQCs, each on concat(x_t, h_{t-1}) ────────────
            ctx = input_size + hidden_size
            self.entry_q = nn.Linear(ctx, n_qubits)
            self.entry_k = nn.Linear(ctx, n_qubits)
            self.entry_v = nn.Linear(ctx, n_qubits)
            self.exit_q  = nn.Linear(n_qubits, hidden_size)
            self.exit_k  = nn.Linear(n_qubits, hidden_size)
            self.exit_v  = nn.Linear(n_qubits, hidden_size)
            self.beta_proj = nn.Linear(input_size, 1)
            self.norm      = nn.LayerNorm(hidden_size)

        # ── VQC(s) — identical circuit to QSSM / cQLSTM ──────────────────
        weight_shapes = {"weights": (n_qlayers, self.n_vrotations, n_qubits)}
        wires = list(range(n_qubits))

        def _make_vqc(dev):
            @qml.qnode(dev, interface="torch")
            def _qnode(inputs, weights):
                features  = inputs.transpose(1, 0)
                ry_params = [torch.arctan(f)    for f in features]
                rz_params = [torch.arctan(f**2) for f in features]
                for i in range(n_qubits):
                    qml.Hadamard(wires=wires[i])
                    qml.RY(ry_params[i], wires=wires[i])
                    qml.RZ(rz_params[i], wires=wires[i])
                qml.layer(self._ansatz, n_qlayers, weights, wires_type=wires)
                return [qml.expval(qml.PauliZ(wires=i)) for i in wires]
            return qml.qnn.TorchLayer(_qnode, weight_shapes)

        if shared_vqc:
            self.VQC = _make_vqc(qml.device(backend, wires=n_qubits))
        else:
            self.VQC_q = _make_vqc(qml.device(backend, wires=n_qubits))
            self.VQC_k = _make_vqc(qml.device(backend, wires=n_qubits))
            self.VQC_v = _make_vqc(qml.device(backend, wires=n_qubits))

    def _ansatz(self, params, wires_type):
        for k in range(self.n_esteps):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % self.n_qubits]])
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def _vqc_proj(self, combined, entry, vqc, exit_):
        """entry(combined) → VQC residual → exit."""
        y = entry(combined)           # [B, n_qubits]
        return exit_(y + vqc(y))      # [B, hidden_size]

    def _forward_legacy(self, x):
        """Shared-VQC path — loads old checkpoints, no recurrence."""
        B, T, _ = x.shape
        x_scaled = x * self.feature_weighting.exp()
        x_flat   = x_scaled.reshape(B * T, -1)

        e       = self.entry(x_flat)
        vqc_out = e + self.VQC(e)
        Q = F.normalize(self.exit_q(vqc_out).reshape(B, T, -1), dim=-1)
        K = F.normalize(self.exit_k(vqc_out).reshape(B, T, -1), dim=-1)
        V = self.exit_v(vqc_out).reshape(B, T, -1) / self.alpha
        beta = torch.sigmoid(self.beta_proj(x_flat)).reshape(B, T, 1)

        raw = chunk_batched_delta_rule_forward(Q, K, V, beta, T)
        return self.norm(self.proj_out(raw)), None

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Parameters
        ----------
        x      : [B, T, input_size]
        hidden : optional h_t  [B, hidden_size]

        Returns
        -------
        outputs : [B, T, hidden_size]
        h_t     : [B, hidden_size]
        """
        if self.shared_vqc:
            return self._forward_legacy(x)

        B, T, _ = x.shape
        d = self.hidden_size

        h_t = (torch.zeros(B, d, device=x.device)
               if hidden is None else hidden)
        S   = torch.zeros(B, d, d, device=x.device)

        x_scaled = x * self.feature_weighting.exp()   # [B, T, input_size]

        outputs = []
        for t in range(T):
            x_t = x_scaled[:, t, :]                   # [B, input_size]

            # each VQC sees current input + previous hidden state
            combined = torch.cat((x_t, h_t), dim=1)  # [B, input+hidden]

            q_t = F.normalize(
                self._vqc_proj(combined, self.entry_q, self.VQC_q, self.exit_q),
                dim=-1)                                # [B, d]
            k_t = F.normalize(
                self._vqc_proj(combined, self.entry_k, self.VQC_k, self.exit_k),
                dim=-1)                                # [B, d]
            v_t = (self._vqc_proj(combined, self.entry_v, self.VQC_v, self.exit_v)
                   / self.alpha)                       # [B, d]

            beta_t = torch.sigmoid(self.beta_proj(x_t))   # [B, 1]

            h_t, S = _delta_step(q_t, k_t, v_t, beta_t, S)
            outputs.append(h_t)

        outputs = self.norm(torch.stack(outputs, dim=1))   # [B, T, d]
        return outputs, h_t
