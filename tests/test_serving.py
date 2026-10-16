import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from moelab.config import ModelConfig  # noqa: E402
from moelab.model.transformer import Transformer  # noqa: E402
from moelab.serving.api import app  # noqa: E402
from moelab.tokenizer.bpe import BPETokenizer  # noqa: E402


def test_readiness_without_checkpoint(monkeypatch):
    monkeypatch.delenv("MOELAB_CHECKPOINT", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.get("/ready").status_code == 503
        assert client.post("/generate", json={"prompt": "hi"}).status_code == 503


def test_serving_checkpoint(tmp_path, monkeypatch):
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    model = Transformer(cfg)
    path = tmp_path / "last.pt"
    torch.save({"model_config": cfg.model_dump(), "model": model.state_dict()}, path)
    BPETokenizer().save(tmp_path / "tokenizer.json")
    monkeypatch.setenv("MOELAB_CHECKPOINT", str(path))
    monkeypatch.setenv("MOELAB_DEVICE", "cpu")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/generate", json={"prompt": "Hi", "max_new_tokens": 3, "temperature": 0}
        )
        assert response.status_code == 200
        assert response.json()["generated_tokens"] == 3
        assert response.json()["text"].startswith("Hi")
        assert client.post("/generate", json={"prompt": "a" * 33}).status_code == 422
        assert client.get("/metrics").status_code == 200


def test_api_reports_eos_and_actual_token_count(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import moelab.serving.api as api

    class EndModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.cfg = SimpleNamespace(vocab_size=257, max_seq_len=32)

        def forward(self, idx):
            logits = torch.zeros((*idx.shape, 257))
            logits[..., 256] = 100
            return logits, None

    monkeypatch.setattr(api, "load_checkpoint", lambda *args: (EndModel(), {}))
    BPETokenizer(257, add_eos=True).save(tmp_path / "tokenizer.json")
    monkeypatch.setenv("MOELAB_CHECKPOINT", str(tmp_path / "last.pt"))
    with TestClient(app) as client:
        response = client.post(
            "/generate", json={"prompt": "Hi", "max_new_tokens": 10, "temperature": 0}
        )
        assert response.status_code == 200
        assert response.json() == {"text": "Hi", "generated_tokens": 1, "finish_reason": "eos"}
