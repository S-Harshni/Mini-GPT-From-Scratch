"""Train the tokenizer and the model, compare with n-gram baselines, and export results for the demo page.

    python train.py            # about ten minutes on a laptop CPU; writes out/ and docs/data.json
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from minigpt.model import GPT, Config, generate  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

SEED, STEPS, BATCH, EVAL_EVERY, EVAL_BATCHES = 1337, 5000, 32, 250, 40
LR, WARMUP, WEIGHT_DECAY = 1e-3, 200, 0.1
PROMPT = "ROMEO:"


def batch(data: torch.Tensor, context: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    start = torch.randint(len(data) - context - 1, (BATCH,), generator=generator)
    return (torch.stack([data[i:i + context] for i in start]), torch.stack([data[i + 1:i + context + 1] for i in start]))


@torch.no_grad()
def evaluate(model: GPT, data: torch.Tensor, generator: torch.Generator) -> float:
    model.eval()
    losses = [model(*batch(data, model.cfg.context, generator))[1].item() for _ in range(EVAL_BATCHES)]
    model.train()
    return sum(losses) / len(losses)


def ngram_baselines(train: list[int], val: list[int], vocab: int) -> dict:
    """Cross-entropy (nats per token) of add-one-smoothed unigram and bigram models on the validation tokens."""
    uni, bi = Counter(train), Counter(zip(train, train[1:], strict=False))
    unigram = -sum(math.log((uni[t] + 1) / (len(train) + vocab)) for t in val) / len(val)
    bigram = -sum(math.log((bi[(a, b)] + 1) / (uni[a] + vocab)) for a, b in zip(val, val[1:], strict=False)) / (len(val) - 1)
    return {"unigram": unigram, "bigram": bigram}


def distinct(ids: list[int], n: int = 2) -> float:
    grams = list(zip(*(ids[i:] for i in range(n)), strict=False))
    return len(set(grams)) / max(len(grams), 1)


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    text = (ROOT / "data" / "tinyshakespeare.txt").read_text()
    cfg = Config()
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    tok = BPETokenizer.train(text[: int(0.9 * len(text))], cfg.vocab_size)
    tok.save(out / "tokenizer.json")
    ids = tok.encode(text)
    split = len(tok.encode(text[: int(0.9 * len(text))]))
    train_ids, val_ids = torch.tensor(ids[:split]), torch.tensor(ids[split:])
    print(f"tokenizer: {tok.size} tokens, {len(text) / len(ids):.2f} characters per token, {time.time() - t0:.0f}s")

    model = GPT(cfg)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": rest, "weight_decay": 0.0}], lr=LR)
    g_train, g_eval = torch.Generator().manual_seed(SEED), torch.Generator().manual_seed(SEED + 1)
    curve, best = [], float("inf")
    t0 = time.time()
    for step in range(STEPS + 1):
        lr = LR * step / WARMUP if step < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
        for group in optim.param_groups:
            group["lr"] = lr
        if step % EVAL_EVERY == 0:
            g_eval.manual_seed(SEED + 1)   # same validation batches every time
            point = {"step": step, "train": evaluate(model, train_ids, torch.Generator().manual_seed(SEED + 2)),
                     "val": evaluate(model, val_ids, g_eval)}
            curve.append(point)
            print(point, f"{time.time() - t0:.0f}s", flush=True)
            if point["val"] < best:
                best = point["val"]
                torch.save(model.state_dict(), out / "model.pt")
        if step == STEPS:
            break
        _, loss = model(*batch(train_ids, cfg.context, g_train))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    train_seconds = time.time() - t0
    model.load_state_dict(torch.load(out / "model.pt"))
    best_point = min(curve, key=lambda c: c["val"])
    baselines = ngram_baselines(train_ids.tolist(), val_ids.tolist(), cfg.vocab_size)

    prompt = torch.tensor([tok.encode(PROMPT)])
    samples = []
    for name, kw in [("Greedy (temperature 0)", {"temperature": 0}), ("Temperature 0.5", {"temperature": 0.5}),
                     ("Temperature 0.8, top-p 0.9", {"temperature": 0.8, "top_p": 0.9}), ("Temperature 1.0", {"temperature": 1.0}),
                     ("Temperature 1.5", {"temperature": 1.5})]:
        rates = []
        for seed in range(5):
            got = generate(model, prompt, 150, generator=torch.Generator().manual_seed(seed), **kw)[0].tolist()
            rates.append(distinct(got[prompt.size(1):]))
            if seed == 0:
                first = tok.decode(got)
        samples.append({"setting": name, "text": first, "distinct_2": round(sum(rates) / len(rates), 4)})

    sentence = "First Citizen:\nWe are accounted poor citizens, the patricians good."
    s_ids = tok.encode(sentence)
    model.eval()
    with torch.no_grad():
        model(torch.tensor([s_ids]))
    attention = [[[[round(v, 3) for v in row] for row in head] for head in block.attention.last_weights[0].tolist()]
                 for block in model.blocks]
    data = {
        "config": cfg.__dict__, "parameters": model.parameters_count(),
        "training": {"steps": STEPS, "batch": BATCH, "seconds": round(train_seconds), "learning_rate": LR, "device": "cpu"},
        "corpus": {"name": "Tiny Shakespeare", "characters": len(text), "tokens": len(ids),
                   "characters_per_token": round(len(text) / len(ids), 3), "validation_tokens": len(val_ids)},
        "curve": [{k: round(v, 4) if k != "step" else v for k, v in c.items()} for c in curve],
        "best": {"step": best_point["step"], "val_loss": round(best_point["val"], 4), "perplexity": round(math.exp(best_point["val"]), 2)},
        "baselines": {k: {"loss": round(v, 4), "perplexity": round(math.exp(v), 2)} for k, v in baselines.items()},
        "samples": samples,
        "attention": {"tokens": [tok.decode([i]) for i in s_ids], "weights": attention},
        "merges": tok.merges,
    }
    ablations = ROOT / "results" / "ablations.json"            # written by ablate.py; kept on the page across retrains
    if ablations.exists():
        data["ablations"] = json.loads(ablations.read_text())
    (ROOT / "docs" / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps({k: data[k] for k in ("parameters", "training", "corpus", "best", "baselines")}, indent=1))
    for s in samples:
        print(s["setting"], s["distinct_2"], repr(s["text"][:120]))


if __name__ == "__main__":
    main()
