"""JEPA predictor — a small transformer that predicts target token latents.

Given the context token embeddings (from the student LitePT encoder) and the
3D positions of the masked target groups, the predictor outputs the predicted
target latents. Mirrors Point-JEPA's `TransformerPredictor`: a narrow
transformer with a learnable mask token, working in a reduced `predictor_dim`
and projecting back to the encoder dimension.

Operates one sample at a time (token counts are O(10^2-10^3) and the per-sample
group count is ragged), so no padding / attention mask is required.
"""

import torch
import torch.nn as nn


class _Block(nn.Module):
    """Pre-norm transformer block (self-attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class JEPAPredictor(nn.Module):
    """Predict target-group latents from context-group latents + target poses.

    Args:
        encoder_dim:   feature dim produced by the LitePT backbone.
        predictor_dim: internal (narrow) transformer width.
        depth:         number of transformer blocks.
        num_heads:     attention heads (must divide `predictor_dim`).
        mlp_ratio:     MLP expansion ratio.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int = 192,
                 depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        assert predictor_dim % num_heads == 0, \
            "predictor_dim must be divisible by num_heads"
        self.enc_to_pred = nn.Linear(encoder_dim, predictor_dim)
        # learnable positional encoder for 3D group centers
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, predictor_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.pred_to_enc = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, context_tokens: torch.Tensor,
                context_centers: torch.Tensor,
                target_centers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_tokens:  (Nc, encoder_dim) student latents of context groups.
            context_centers: (Nc, 3) context group centers.
            target_centers:  (Nt, 3) target group centers to predict.
        Returns:
            (Nt, encoder_dim) predicted target latents.
        """
        # normalise positions w.r.t. the context extent for scale invariance
        ref_mean = context_centers.mean(dim=0, keepdim=True)
        ref_scale = context_centers.std().clamp(min=1e-3)
        ctx_pos = self.pos_embed((context_centers - ref_mean) / ref_scale)
        tgt_pos = self.pos_embed((target_centers - ref_mean) / ref_scale)

        x_ctx = self.enc_to_pred(context_tokens) + ctx_pos          # (Nc, P)
        x_tgt = self.mask_token + tgt_pos                            # (Nt, P)

        n_ctx = x_ctx.shape[0]
        x = torch.cat([x_ctx, x_tgt], dim=0).unsqueeze(0)            # (1, Nc+Nt, P)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[0, n_ctx:])                                  # (Nt, P)
        return self.pred_to_enc(x)                                   # (Nt, enc_dim)
