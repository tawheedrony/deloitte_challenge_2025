import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── import chunk kernel from local library ────────────────────────────────
_LIBS = Path(__file__).parent.parent.parent / "libs" / "deltanet"
sys.path.insert(0, str(_LIBS))
from deltanet import chunk_batched_delta_rule_forward  # noqa: E402


class DeltaNet(nn.Module):
    """
    Classical DeltaNet for sequence modelling.

    Implements the delta rule directly (rather than via DeltaBlock) so that
    L2-normalised Q/K can be applied before the chunk kernel — this is the
    critical stability fix absent from the reference DeltaBlock.

    Architecture
    ------------
        x_scaled  =  x * exp(feature_weighting)         [B, T, input_size]
        Q         =  normalize( Wq(x_scaled) )           [B, T, d]
        K         =  normalize( Wk(x_scaled) )           [B, T, d]    ← key fix
        V         =  Wv(x_scaled) / alpha                [B, T, d]
        beta      =  alpha * sigmoid( beta_proj(x) )     [B, T, 1]

        raw       =  chunk_delta_rule(Q, K, V, beta)     [B, T, d]
        output    =  LayerNorm( proj_out(raw) )           [B, T, hidden_size]

    Why Q/K normalisation matters
    ------------------------------
    The chunk kernel builds T = -(K_beta @ K^T) and applies forward
    substitution to invert (I - lower_tri(T)).  Without normalisation
    ||k_i · k_j|| is unbounded, so T can have entries >> 1 and the
    series diverges.  Normalising K to unit norm bounds each entry of T to
    (-1, 0), guaranteeing convergence of the forward substitution.

    Parameters
    ----------
    input_size  : int   — number of input features
    hidden_size : int   — output dimension
    expand      : int   — Q/K/V inner dimension = hidden_size * expand
    neg_eigen   : bool  — allow negative eigenvalues (alpha=2 scaling)
    chunk_size  : int   — chunk size (0 = full sequence)

    Interface
    ---------
    forward(x, hidden=None)
        x       : [B, T, input_size]
        returns : (outputs, None)
                  outputs : [B, T, hidden_size]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        expand: int = 1,
        neg_eigen: bool = False,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size  = chunk_size
        self.alpha       = 2 if neg_eigen else 1
        d                = hidden_size * expand

        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))

        self.Wq        = nn.Linear(input_size, d)
        self.Wk        = nn.Linear(input_size, d)
        self.Wv        = nn.Linear(input_size, d)
        self.beta_proj = nn.Linear(input_size, 1)
        self.sigma     = nn.Sigmoid()
        self.proj_out  = nn.Linear(d, hidden_size)
        self.norm      = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Parameters
        ----------
        x      : [B, T, input_size]
        hidden : unused — kept for API compatibility

        Returns
        -------
        outputs : [B, T, hidden_size]
        None    : placeholder hidden state
        """
        B, T, _ = x.shape

        x_scaled = x * self.feature_weighting.exp()          # [B, T, input_size]

        # L2-normalise Q and K — essential for stable T-matrix computation
        Q    = F.normalize(self.Wq(x_scaled), dim=-1)        # [B, T, d]
        K    = F.normalize(self.Wk(x_scaled), dim=-1)        # [B, T, d]
        V    = self.Wv(x_scaled) / self.alpha                 # [B, T, d]
        beta = self.alpha * self.sigma(self.beta_proj(x_scaled))  # [B, T, 1]

        chunk = T if self.chunk_size == 0 else self.chunk_size
        raw   = chunk_batched_delta_rule_forward(Q, K, V, beta, chunk)  # [B, T, d]

        out   = self.norm(self.proj_out(raw))                 # [B, T, hidden_size]
        return out, None
