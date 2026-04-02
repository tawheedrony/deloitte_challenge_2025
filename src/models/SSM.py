import torch
import torch.nn as nn


class SSM(nn.Module):
    """
    Vanilla (classical) State Space Model.

    Classical counterpart to QuantumSSM — identical selective recurrence
    and feature weighting, but the quantum VQC+entry+exit stack is replaced
    by a single linear projection.

    Recurrence
    ----------
        combined_t  =  concat(x_t, h_t)                       [B, input+hidden]
        out_t       =  fc(combined_t)                          [B, hidden*2]

        g_raw, delta_raw  =  out_t.chunk(2)
        g_t     =  clamp( sigmoid(g_raw), g_min, g_max )      [B, hidden]
        delta_t =  tanh( delta_raw )                           [B, hidden]

        h_t  =  (1 - g_t) * h_{t-1}  +  g_t * delta_t

    Comparison with QuantumSSM
    --------------------------
    | Aspect          | SSM (this)             | QuantumSSM              |
    |-----------------|------------------------|-------------------------|
    | Gate projection | Linear(input+h, h*2)   | entry→VQC→exit residual |
    | Quantum circuit | none                   | 4-qubit VQC             |
    | Parameters      | (input+hidden)*hidden*2| entry + VQC + exit      |
    | Speed           | fast                   | slow (sim overhead)     |
    | Gate structure  | g, delta               | g, delta                |
    | Feature weight  | learnable exp scale    | learnable exp scale     |

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
        hidden_size: int = 64,
        g_min: float = 0.05,
        g_max: float = 0.95,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.g_min       = g_min
        self.g_max       = g_max

        # learnable per-feature scaling (same as cQLSTM / QSSM)
        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))

        # single linear replaces entry + VQC + exit
        self.fc = nn.Linear(input_size + hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, hidden=None):
        B, T, _ = x.shape

        h_t = (
            torch.zeros(B, self.hidden_size, device=x.device)
            if hidden is None
            else hidden
        )

        x = x * self.feature_weighting.exp()

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, h_t), dim=1)        # [B, input+hidden]
            out      = self.fc(combined)                    # [B, hidden*2]

            g_raw, delta_raw = out.chunk(2, dim=1)

            g_t     = torch.clamp(torch.sigmoid(g_raw), self.g_min, self.g_max)
            delta_t = torch.tanh(delta_raw)

            h_t = (1.0 - g_t) * h_t + g_t * delta_t       # [B, hidden]
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t             # [B, T, hidden], [B, hidden]
