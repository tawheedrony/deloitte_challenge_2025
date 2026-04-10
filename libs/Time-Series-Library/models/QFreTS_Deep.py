"""
Quantum-Enhanced FreTS with Deep Circuit Integration (QFreTS_Deep).

Unlike QFreTS (which wraps FreTS output with a post-hoc quantum block),
this model integrates quantum circuits at three internal computation points:

1. **Quantum Channel Gate** — replaces the classical MLP_channel with a
   quantum-enhanced squeeze-and-excite gate.  A global pool over time and
   embedding yields a compact per-channel summary that is processed through
   a VQC; the resulting gate modulates cross-variable information flow via
   quantum entanglement between channel qubits.

2. **Quantum Embedding Refinement** — implements "data re-uploading across
   layers": after the classical temporal-frequency MLP, embeddings are pooled
   over time and refined through a second quantum circuit, broadcasting the
   quantum-enhanced representation back across the time axis.

3. **Quantum Latent Head** — the prediction FC pathway is split into two
   halves with a quantum residual block in between, letting the VQC operate
   on the compact latent representation just before the final linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Quantum_Hybrid import QuantumResidualBlock


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.feature_size = configs.enc_in
        self.embed_size = 128
        self.hidden_size = 256
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        # ── token embedding (same as FreTS) ─────────────────────────
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        # ── temporal frequency MLP (classical, same as FreTS) ───────
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        # ── quantum hyper-parameters ────────────────────────────────
        n_qubits = getattr(configs, "n_qubits", 4)
        n_qlayers = getattr(configs, "n_qlayers", 1)
        n_esteps = getattr(configs, "n_esteps", 1)
        data_reupload = getattr(configs, "data_reupload", False)
        backend = getattr(configs, "quantum_backend", "default.qubit")
        qkw = dict(
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            n_esteps=n_esteps,
            data_reupload=data_reupload,
            backend=backend,
        )

        # ── (1) quantum channel gate ────────────────────────────────
        # Replaces classical MLP_channel.  Global pool → VQC → sigmoid gate.
        self.q_channel = QuantumResidualBlock(self.feature_size, self.feature_size, **qkw)

        # ── (2) quantum embedding refinement (re-uploading) ─────────
        # Applies a second quantum pass after temporal-frequency processing.
        self.q_embed = QuantumResidualBlock(self.embed_size, self.embed_size, **qkw)

        # ── (3) quantum latent head ─────────────────────────────────
        # FC-in → quantum → FC-out  (quantum in prediction pathway).
        self.fc_in = nn.Linear(self.seq_len * self.embed_size, self.hidden_size)
        self.q_latent = QuantumResidualBlock(self.hidden_size, self.hidden_size, **qkw)
        self.fc_out = nn.Linear(self.hidden_size, self.pred_len)

    # ── helpers (same as FreTS) ──────────────────────────────────────

    def tokenEmb(self, x):
        # x: [B, T, N] → [B, N, T, D]
        x = x.permute(0, 2, 1).unsqueeze(3)
        return x * self.embeddings

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        y = self._FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        return torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")

    def _FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, r)
            - torch.einsum("bijd,dd->bijd", x.imag, i)
            + rb
        )
        o1_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, r)
            + torch.einsum("bijd,dd->bijd", x.real, i)
            + ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    # ── quantum integration stages ───────────────────────────────────

    def _quantum_channel_gate(self, x):
        """Squeeze-and-excite style channel gating via VQC.

        Pool over time & embedding → quantum → sigmoid → modulate channels.
        The VQC sees all channels simultaneously; entanglement captures
        cross-variable correlations that inform the gating decision.
        Circuit evaluations: B  (one per batch element).
        """
        gate = x.mean(dim=(2, 3))                    # [B, N]
        gate = self.q_channel(gate)                   # [B, N]
        gate = torch.sigmoid(gate)                    # [B, N]
        return x * gate.unsqueeze(-1).unsqueeze(-1)   # [B, N, T, D]

    def _quantum_embed_refine(self, x):
        """Data re-uploading: second quantum pass on time-pooled embeddings.

        Pool over time → quantum on embedding dim → broadcast-add back.
        Circuit evaluations: B * N.
        """
        B, N, T, D = x.shape
        pooled = x.mean(dim=2)                       # [B, N, D]
        refined = self.q_embed(pooled.reshape(-1, D)) # [B*N, D]
        return x + refined.reshape(B, N, 1, D)        # broadcast over T

    # ── forward ──────────────────────────────────────────────────────

    def forecast(self, x_enc):
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)                     # [B, N, T, D]
        bias = x

        # Stage 1: quantum channel gating (cross-variable mixing)
        x = self._quantum_channel_gate(x)

        # Stage 2: classical temporal-frequency learning
        x = self.MLP_temporal(x, B, N, T)

        # Stage 3: quantum embedding refinement (data re-uploading)
        x = self._quantum_embed_refine(x)

        x = x + bias                                  # residual

        # Stage 4: quantum latent prediction head
        x = x.reshape(B, N, -1)                       # [B, N, T*D]
        x = F.leaky_relu(self.fc_in(x))               # [B, N, hidden]
        x = self.q_latent(x)                           # [B, N, hidden]
        x = self.fc_out(x).permute(0, 2, 1)           # [B, pred_len, N]
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        raise ValueError("Only forecast tasks implemented")
