import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from moelab.config import ModelConfig  # noqa: E402
from moelab.model.transformer import Transformer  # noqa: E402
from moelab.serving.api import app  # noqa: E402
from moelab.tokenizer.bpe import BPETokenizer  # noqa: E402


@pytest.mark.parametrize("top_k", [None, 1, 2, 4])
def test_real_prompt_routing(tmp_path, monkeypatch, top_k):
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        moe={"n_experts": 4, "top_k": top_k} if top_k else None,
    )
    model = Transformer(cfg)
    path = tmp_path / "last.pt"
    torch.save({"model_config": cfg.model_dump(), "model": model.state_dict()}, path)
    BPETokenizer().save(tmp_path / "tokenizer.json")
    monkeypatch.setenv("MOELAB_CHECKPOINT", str(path))
    with TestClient(app) as client:
        metadata = client.get("/model").json()
        assert metadata["total_parameters"] == model.num_params()
        assert metadata["active_parameters_estimate"] == model.num_params(True)
        response = client.post("/inspect", json={"prompt": "Hi"})
        assert response.status_code == 200
        result = response.json()
        assert len(result["tokens"]) == 2
        assert len(result["layers"]) == (2 if top_k else 0)
        for layer in result["layers"]:
            probs = torch.tensor(layer["probabilities"])
            weights, selected = probs.topk(top_k, dim=-1)
            if top_k > 1:
                weights /= weights.sum(-1, keepdim=True)
            assert layer["selected"] == selected.tolist()
            assert torch.allclose(torch.tensor(layer["weights"]), weights)
            assert torch.allclose(probs.sum(-1), torch.ones(2))
            assert sum(layer["counts"]) == 2 * top_k
        # Full output can exceed the context: retain one row per token, and
        # verify its last row against a direct inspection of the final window.
        text = "Hi" + "a" * 33
        long_response = client.post(
            "/inspect",
            json={
                "prompt": text,
                "allow_long_text": True,
            },
        )
        assert long_response.status_code == 200
        long_result = long_response.json()
        assert long_result["text"] == text and len(long_result["tokens"]) == 35
        tail = client.post("/inspect", json={"prompt": text[-32:]}).json()
        for actual, expected in zip(long_result["layers"], tail["layers"], strict=True):
            assert len(actual["selected"]) == 35
            assert len(actual["weights"]) == 35
            assert sum(actual["counts"]) == 35 * top_k
            assert torch.allclose(
                torch.tensor(actual["probabilities"][-1]),
                torch.tensor(expected["probabilities"][-1]),
            )
        assert (
            client.post(
                "/inspect",
                json={
                    "prompt": "a" * 2049,
                    "allow_long_text": True,
                },
            ).status_code
            == 422
        )
        # Hooks must not leak into generation or later requests.
        if top_k:
            assert not app.state.model.blocks[0].ffn.router._forward_hooks
        assert client.post("/inspect", json={"prompt": " "}).status_code == 422
        assert client.post("/inspect", json={"prompt": "a" * 33}).status_code == 422
        assert (
            client.post("/generate", json={"prompt": "Hi", "max_new_tokens": 1}).status_code == 200
        )


def test_workspace_unavailable_and_archives(tmp_path, monkeypatch):
    monkeypatch.delenv("MOELAB_CHECKPOINT", raising=False)
    monkeypatch.setenv("MOELAB_REPORTS_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/model").status_code == 503
        assert client.post("/inspect", json={"prompt": "Hi"}).status_code == 503
        assert client.get("/experiments").json()["runs"] == []
        monkeypatch.setenv("MOELAB_REPORTS_DIR", "reports")
        data = client.get("/experiments").json()
        assert len(data["runs"]) == 6
        assert {run["model"] for run in data["runs"]} == {"dense", "moe"}
        assert all(len(run["validation_curve"]) == 11 for run in data["runs"])


def test_training_history_handles_partial_write_and_preserves_validation(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("MOELAB_TRAINING_DIR", str(tmp_path))
    monkeypatch.delenv("MOELAB_CHECKPOINT", raising=False)
    with TestClient(app) as client:
        assert client.get("/training").json()["points"] == []
        rows = [{"step": i, "train_loss": 3.0, "total_steps": 20000} for i in range(2001)]
        rows[501]["val_ce"] = 2.5
        (tmp_path / "metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + '\n{"step":', encoding="utf-8"
        )
        result = client.get("/training").json()
        assert result["step"] == 2000 and result["total_steps"] == 20000
        assert any(p["step"] == 501 and p["val_ce"] == 2.5 for p in result["points"])
        assert result["points"][-1]["step"] == 2000
        assert len(result["points"]) < 1100
