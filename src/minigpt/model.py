"""A decoder-only transformer. Attention is written out by hand so each step is visible."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 1024
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.context, cfg.context)).bool(), persistent=False)
        self.last_weights: torch.Tensor | None = None   # kept for the attention visualisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in self.qkv(x).chunk(3, dim=-1))
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))           # (b, heads, t, t)
        scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))     # a token cannot see later tokens
        weights = scores.softmax(dim=-1)
        self.last_weights = weights.detach()
        y = (self.drop(weights) @ v).transpose(1, 2).reshape(b, t, c)
        return self.drop(self.out(y))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attention = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width),
                                 nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))      # pre-norm residual blocks
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positions = nn.Embedding(cfg.context, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.head.weight = self.tokens.weight      # weight tying: input and output embeddings are shared
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        t = ids.size(1)
        x = self.drop(self.tokens(ids) + self.positions(torch.arange(t, device=ids.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Top-k keeps the k likeliest tokens; top-p (nucleus) keeps the smallest set whose probability reaches p."""
    logits = logits.clone()
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p is not None:
        ordered, index = torch.sort(logits, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        drop = cumulative - ordered.softmax(dim=-1) >= top_p     # mass before this token already reaches p
        logits[torch.zeros_like(drop).scatter(-1, index, drop)] = float("-inf")   # back to vocabulary order
    return logits


@torch.no_grad()
def generate(model: GPT, ids: torch.Tensor, new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    model.eval()
    for _ in range(new_tokens):
        logits, _ = model(ids[:, -model.cfg.context:])
        logits = logits[:, -1]
        if temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        ids = torch.cat([ids, nxt], dim=1)
    return ids
