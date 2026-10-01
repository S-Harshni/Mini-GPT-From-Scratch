import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from minigpt.model import GPT, CausalSelfAttention, Config, filter_logits, generate  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

TEXT = "the cat sat on the mat. the cat ate the rat. the rat sat on the cat.\n" * 20
SMALL = Config(vocab_size=300, context=16, layers=2, heads=2, width=32, dropout=0.0)


def test_bpe_round_trip_and_compression():
    tok = BPETokenizer.train(TEXT, 300)
    for sample in (TEXT, "unseen words: zebra, façade, 日本", "  leading and trailing  ", ""):
        assert tok.decode(tok.encode(sample)) == sample
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2      # merges shorten the sequence
    assert tok.size == 256 + len(tok.merges) <= 300


def test_bpe_learns_the_most_frequent_pair_first_and_is_deterministic():
    a, b = BPETokenizer.train(TEXT, 280), BPETokenizer.train(TEXT, 280)
    assert a.merges == b.merges
    first = a.vocab[256]
    assert first in (b" t", b"th", b"he", b"at")              # a pair that really is frequent in the text


def test_bpe_save_and_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 280)
    tok.save(tmp_path / "t.json")
    assert BPETokenizer.load(tmp_path / "t.json").encode("the cat") == tok.encode("the cat")


def test_attention_matches_the_reference_implementation():
    torch.manual_seed(0)
    layer = CausalSelfAttention(SMALL).eval()
    x = torch.randn(2, 10, SMALL.width)
    q, k, v = (z.view(2, 10, SMALL.heads, -1).transpose(1, 2) for z in layer.qkv(x).chunk(3, dim=-1))
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, SMALL.width)
    assert torch.allclose(layer(x), layer.out(reference), atol=1e-5)
    assert torch.allclose(layer.last_weights.sum(-1), torch.ones(2, SMALL.heads, 10), atol=1e-5)
    assert layer.last_weights.triu(1).abs().max() == 0        # nothing attends to the future


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    a = torch.randint(0, 300, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 300, (4,))
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5) and not torch.allclose(la[0, 8:], lb[0, 8:])


def test_initial_loss_is_uniform_and_training_reduces_it():
    torch.manual_seed(0)
    model = GPT(SMALL)
    x = torch.randint(0, 300, (8, 16))
    y = torch.roll(x, -1, dims=1)
    start = model(x, y)[1].item()
    assert start == pytest.approx(torch.log(torch.tensor(300.0)).item(), abs=0.3)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = model(x, y)[1]
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert loss.item() < 0.5 * start                           # it can memorise one batch


def test_weight_tying_and_parameter_count():
    model = GPT(SMALL)
    assert model.head.weight is model.tokens.weight
    assert model.parameters_count() == sum(p.numel() for p in set(model.parameters()))


def test_top_k_and_top_p_filters():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.07, 0.03]]))
    assert torch.isfinite(filter_logits(logits, top_k=2)).sum() == 2
    kept = torch.isfinite(filter_logits(logits, top_p=0.7))
    assert kept.tolist() == [[True, True, False, False, False]]    # 0.5 + 0.25 reaches 0.7
    assert torch.isfinite(filter_logits(logits, top_p=0.4)).sum() == 1   # the top token always survives
    shuffled = logits[:, [3, 0, 4, 1, 2]]
    assert torch.isfinite(filter_logits(shuffled, top_p=0.7)).tolist() == [[False, True, False, True, False]]


def test_generation_is_reproducible_and_greedy_is_deterministic():
    torch.manual_seed(0)
    model = GPT(SMALL)
    start = torch.zeros((1, 1), dtype=torch.long)
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    assert torch.equal(generate(model, start, 20, 0.8, generator=g(1)), generate(model, start, 20, 0.8, generator=g(1)))
    assert torch.equal(generate(model, start, 20, temperature=0), generate(model, start, 20, temperature=0))
    assert generate(model, start, 30, 1.0, top_p=0.9, generator=g(2)).shape == (1, 31)   # runs past the context window
