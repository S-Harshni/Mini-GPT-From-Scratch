"""Ablations: what does each part of the model buy? Each variant is trained with the same data, seed and budget.

    python train.py && python ablate.py      # writes results/ablations.json
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from minigpt.model import GPT, Config  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from train import LR, SEED, WEIGHT_DECAY, batch, evaluate  # noqa: E402

STEPS, WARMUP, EVAL_EVERY = 1500, 100, 250
VARIANTS = {
    "Full model (4 layers, 4 heads)": {},
    "No position embeddings": {"use_positions": False},
    "One attention head": {"heads": 1},
    "One layer": {"layers": 1},
    "Context of 16 tokens": {"context": 16},
}


def train(cfg: Config, train_ids: torch.Tensor, val_ids: torch.Tensor) -> dict:
    torch.manual_seed(SEED)
    model = GPT(cfg)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": rest, "weight_decay": 0.0}], lr=LR)
    g_train = torch.Generator().manual_seed(SEED)
    best, t0 = float("inf"), time.time()
    for step in range(STEPS + 1):
        lr = LR * step / WARMUP if step < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
        for group in optim.param_groups:
            group["lr"] = lr
        if step % EVAL_EVERY == 0:
            best = min(best, evaluate(model, val_ids, torch.Generator().manual_seed(SEED + 1)))
        if step == STEPS:
            break
        _, loss = model(*batch(train_ids, cfg.context, g_train))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    return {"parameters": model.parameters_count(), "val_loss": round(best, 4), "perplexity": round(math.exp(best), 2),
            "seconds": round(time.time() - t0)}


def main() -> None:
    torch.set_num_threads(4)
    text = (ROOT / "data" / "tinyshakespeare.txt").read_text()
    tok = BPETokenizer.load(ROOT / "out" / "tokenizer.json")
    ids = tok.encode(text)
    split = len(tok.encode(text[: int(0.9 * len(text))]))
    train_ids, val_ids = torch.tensor(ids[:split]), torch.tensor(ids[split:])
    rows = []
    for name, change in VARIANTS.items():
        row = {"variant": name, **train(Config(**change), train_ids, val_ids)}
        rows.append(row)
        print(row, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ablations.json").write_text(json.dumps({"steps": STEPS, "variants": rows}, indent=1))


if __name__ == "__main__":
    main()
