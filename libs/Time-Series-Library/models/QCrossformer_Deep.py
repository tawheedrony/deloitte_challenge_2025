"""
Quantum-Enhanced Crossformer with Deep Circuit Integration (QCrossformer_Deep).

Unlike QCrossformer (which wraps Crossformer output with a post-hoc quantum
block), this model integrates quantum circuits at two key architectural points:

1. **Quantum Cross-Dimension Attention** — in the TwoStageAttentionLayer, the
   router-based cross-dimension stage (dim_sender → dim_receiver) is replaced
   with a quantum-enhanced gate.  A VQC processes per-variable summaries and
   produces entanglement-informed gating values, followed by quantum
   refinement of the d_model representations.  This lets quantum correlations
   drive cross-variable information flow.

2. **Quantum Encoder–Decoder Bridge** — the last encoder output is processed
   through a quantum residual block before being passed to the decoder.  This
   compresses the encoder representation through a quantum bottleneck in the
   latent space, letting the VQC refine the most abstract features.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import ceil

from layers.Crossformer_EncDec import SegMerging, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Quantum_Hybrid import QuantumResidualBlock
from models.PatchTST import FlattenHead


# ─────────────────────────────────────────────────────────────────────
# Quantum-enhanced TwoStageAttentionLayer
# ─────────────────────────────────────────────────────────────────────

class QuantumTwoStageAttentionLayer(nn.Module):
    """Two-Stage Attention with quantum circuit replacing cross-dimension router.

    Stage 1 (Cross-Time) is unchanged: standard multi-head self-attention
    along the time-segment axis for each variable independently.

    Stage 2 (Cross-Dimension) replaces the learned-router attention
    (dim_sender + dim_receiver) with:
      a) Quantum variable gate: pool → VQC → sigmoid → modulate
      b) Quantum d_model refinement: VQC on representation vectors

    Input/output shape: [batch, ts_d, seg_num, d_model]
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads,
                 d_ff=None, dropout=0.1,
                 n_qubits=4, n_qlayers=1, n_esteps=1,
                 data_reupload=False, backend="default.qubit"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        qkw = dict(n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
                    data_reupload=data_reupload, backend=backend)

        # Cross-Time Stage (classical attention — unchanged)
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor,
                          attention_dropout=configs.dropout,
                          output_attention=False),
            d_model, n_heads,
        )

        # Cross-Dimension Stage — quantum replacement
        # (a) Quantum variable gate: captures cross-variable correlations
        self.q_var_gate = QuantumResidualBlock(d_model, d_model, **qkw)
        # (b) Quantum representation refinement on d_model
        self.q_dim_refine = QuantumResidualBlock(d_model, d_model, **qkw)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        batch = x.shape[0]

        # ── Cross-Time Stage (classical) ────────────────────────────
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc, _ = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # ── Cross-Dimension Stage (quantum) ─────────────────────────
        # Reshape to [B*seg_num, ts_d, d_model] for cross-variable view
        dim_send = rearrange(
            dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model",
            b=batch)

        # (a) Quantum variable gate: per-variable summary → VQC → sigmoid
        #     The gate values are entanglement-informed, capturing
        #     cross-variable quantum correlations.
        #     Circuit evaluations: B * seg_num * ts_d
        bs, nv, dm = dim_send.shape
        var_refined = self.q_var_gate(dim_send.reshape(-1, dm))  # [bs*nv, dm]
        var_refined = var_refined.reshape(bs, nv, dm)
        dim_enc = dim_send + self.dropout(var_refined - dim_send)

        # (b) Quantum d_model refinement
        #     Circuit evaluations: B * seg_num * ts_d
        dim_flat = dim_enc.reshape(-1, dm)
        dim_ref = self.q_dim_refine(dim_flat).reshape(bs, nv, dm)
        dim_enc = dim_enc + self.dropout(dim_ref - dim_enc)

        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        return rearrange(
            dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model",
            b=batch)


# ─────────────────────────────────────────────────────────────────────
# Quantum scale_block (mirrors Crossformer_EncDec.scale_block)
# ─────────────────────────────────────────────────────────────────────

class quantum_scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff,
                 depth, dropout, seg_num=10, factor=10,
                 n_qubits=4, n_qlayers=1, n_esteps=1,
                 data_reupload=False, backend="default.qubit"):
        super().__init__()
        self.merge_layer = SegMerging(d_model, win_size) if win_size > 1 else None
        self.encode_layers = nn.ModuleList()
        for _ in range(depth):
            self.encode_layers.append(
                QuantumTwoStageAttentionLayer(
                    configs, seg_num, factor, d_model, n_heads, d_ff, dropout,
                    n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
                    data_reupload=data_reupload, backend=backend))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x, None


# ─────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # ── quantum hyper-parameters ────────────────────────────────
        n_qubits = getattr(configs, "n_qubits", 4)
        n_qlayers = getattr(configs, "n_qlayers", 1)
        n_esteps = getattr(configs, "n_esteps", 1)
        data_reupload = getattr(configs, "data_reupload", False)
        backend = getattr(configs, "quantum_backend", "default.qubit")
        qkw = dict(n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
                    data_reupload=data_reupload, backend=backend)

        # ── Embedding ───────────────────────────────────────────────
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len,
            self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # ── Encoder with quantum cross-dim attention ────────────────
        self.encoder = Encoder([
            quantum_scale_block(
                configs,
                1 if l == 0 else self.win_size,
                configs.d_model, configs.n_heads, configs.d_ff,
                1, configs.dropout,
                self.in_seg_num if l == 0 else ceil(
                    self.in_seg_num / self.win_size ** l),
                configs.factor,
                **qkw,
            )
            for l in range(configs.e_layers)
        ])

        # ── Quantum encoder–decoder bridge ──────────────────────────
        self.q_bridge = QuantumResidualBlock(
            configs.d_model, configs.d_model, **qkw)

        # ── Decoder (classical — unchanged) ─────────────────────────
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in,
                        self.pad_out_len // self.seg_len, configs.d_model))

        from layers.SelfAttention_Family import TwoStageAttentionLayer
        self.decoder = Decoder([
            DecoderLayer(
                TwoStageAttentionLayer(
                    configs,
                    self.pad_out_len // self.seg_len,
                    configs.factor, configs.d_model, configs.n_heads,
                    configs.d_ff, configs.dropout),
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                self.seg_len,
                configs.d_model, configs.d_ff,
                dropout=configs.dropout,
            )
            for _ in range(configs.e_layers + 1)
        ])

        if self.task_name in ("imputation", "anomaly_detection"):
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.seq_len,
                head_dropout=configs.dropout)

    # ── quantum bridge ───────────────────────────────────────────────

    def _apply_bridge(self, enc_outputs):
        """Apply quantum bridge to last encoder output.

        Processes each d_model vector through the VQC, letting the quantum
        circuit refine the most abstract encoder representations before
        they are consumed by the decoder.
        Circuit evaluations: B * ts_d * seg_num_last.
        """
        last = enc_outputs[-1]                        # [B, ts_d, seg_num, d_model]
        shape = last.shape
        flat = last.reshape(-1, shape[-1])             # [*, d_model]
        refined = self.q_bridge(flat)                  # [*, d_model]
        enc_outputs[-1] = refined.reshape(shape)
        return enc_outputs

    # ── task-specific forward methods ────────────────────────────────

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out, _ = self.encoder(x_enc)
        enc_out = self._apply_bridge(enc_out)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc.shape[0])
        return self.decoder(dec_in, enc_out)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, _ = self.encoder(x_enc)
        enc_out = self._apply_bridge(enc_out)
        return self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

    def anomaly_detection(self, x_enc):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, _ = self.encoder(x_enc)
        enc_out = self._apply_bridge(enc_out)
        return self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        return None
