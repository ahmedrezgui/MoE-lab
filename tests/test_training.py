import numpy as np
import pytest

torch = pytest.importorskip("torch")
from moelab.config import ModelConfig, TrainingConfig  # noqa: E402
from moelab.inference.generate import generate  # noqa: E402
from moelab.tokenizer.bpe import BPETokenizer  # noqa: E402
from moelab.training.loop import load_checkpoint, train  # noqa: E402


@pytest.mark.parametrize("legacy", [False, True])
def test_training_resume_equals_uninterrupted(tmp_path, legacy):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    for name in ("train", "val"):
        np.tile(np.arange(40, dtype="<u2"), 10).tofile(data / f"ids_{name}.bin")
    model_cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.1,
        moe={"n_experts": 4, "top_k": 2},
    )
    training_cfg = TrainingConfig(
        lr=0.003, warmup_steps=1, max_steps=4, batch_size=2, precision="fp32", grad_accum_steps=2
    )
    full = train(model_cfg, training_cfg, data_dir=data, output_dir=tmp_path / "full")
    train(model_cfg, training_cfg, max_steps=2, data_dir=data, output_dir=tmp_path / "split")
    if legacy:
        checkpoint_path = tmp_path / "split/last.pt"
        old = torch.load(checkpoint_path, weights_only=True)
        old["model_config"].pop("ffn_hidden_dim")
        torch.save(old, checkpoint_path)
    resumed = train(
        model_cfg,
        training_cfg,
        data_dir=data,
        output_dir=tmp_path / "split",
        resume=tmp_path / "split/last.pt",
    )
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    loaded, checkpoint = load_checkpoint(tmp_path / "split/last.pt")
    assert checkpoint["step"] == 4
    x = torch.tensor([[1, 2, 3]])
    loaded.train()
    result = generate(loaded, x, 4, temperature=0)
    assert result.shape == (1, 7) and loaded.training
    assert torch.equal(result, generate(resumed, x, 4, temperature=0))


def test_sampler_greedy_and_context_window():
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = type("Config", (), {"max_seq_len": 3})()

        def forward(self, x):
            assert x.size(1) <= 3
            logits = torch.zeros(x.size(0), x.size(1), 5)
            logits[..., 2] = 10
            return logits, None

    model = ConstantModel()
    result = generate(model, torch.tensor([[1, 0]]), 5, temperature=0)
    assert result.tolist() == [[1, 0, 2, 2, 2, 2, 2]]
    for kwargs in [{"top_k": 0}, {"top_p": 0}, {"temperature": -1}]:
        with pytest.raises(ValueError):
            generate(model, torch.tensor([[1]]), 1, **kwargs)


def test_training_uses_effective_bpe_vocabulary(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    tokenizer = BPETokenizer(512)
    tokenizer.train("abab")
    assert tokenizer.vocab_size == 257
    tokenizer.save(data / "tokenizer.json")
    for split in ("train", "val"):
        ids = tokenizer.encode("abab abcdefghijklmnopqrstuvwxyz " * 10)
        np.asarray(ids, dtype="<u2").tofile(data / f"ids_{split}.bin")
    model_cfg = ModelConfig(
        vocab_size=257, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    training_cfg = TrainingConfig(
        lr=0.001, warmup_steps=0, max_steps=1, batch_size=1, precision="fp32"
    )
    inflated = model_cfg.model_copy(update={"vocab_size": 512})
    with pytest.raises(ValueError, match="actual vocabulary size 257"):
        train(inflated, training_cfg, data_dir=data, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    train(model_cfg, training_cfg, data_dir=data, output_dir=tmp_path / "valid")
    model, _ = load_checkpoint(tmp_path / "valid/best.pt")
    result = generate(model, torch.tensor([[97, 98]]), 8, temperature=0)
    assert result.max() < tokenizer.vocab_size
    assert isinstance(tokenizer.decode(result[0].tolist()), str)
