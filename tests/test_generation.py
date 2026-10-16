from types import SimpleNamespace

import pytest
import torch

from moelab.inference.generate import generate


class ScriptedModel(torch.nn.Module):
    def __init__(self, schedule, vocab_size=5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cfg = SimpleNamespace(vocab_size=vocab_size, max_seq_len=32)
        self.schedule = schedule
        self.calls = 0

    def forward(self, idx):
        choices = self.schedule[min(self.calls, len(self.schedule) - 1)]
        self.calls += 1
        logits = torch.full((*idx.shape, self.cfg.vocab_size), -100.0, device=idx.device)
        for row, token in enumerate(choices):
            logits[row, -1, token] = 100
        return logits, None


@pytest.mark.parametrize("temperature", [0, 1])
def test_batch_stops_independently_and_restores_training(temperature):
    model = ScriptedModel([[4, 2], [1, 3], [1, 4]])
    result = generate(
        model, torch.tensor([[0], [0]]), 10, temperature=temperature, top_k=1, eos_token_id=4
    )
    assert result.tolist() == [[0, 4, 4, 4], [0, 2, 3, 4]]
    assert model.calls == 3 and model.training


def test_prompt_eos_does_not_stop_and_legacy_generation_has_fixed_length():
    model = ScriptedModel([[2], [4]])
    model.eval()
    result = generate(model, torch.tensor([[4]]), 10, temperature=0, eos_token_id=4)
    assert result.tolist() == [[4, 2, 4]] and not model.training
    model = ScriptedModel([[4]])
    assert generate(model, torch.tensor([[0]]), 3, temperature=0).tolist() == [[0, 4, 4, 4]]
    assert model.calls == 3


def test_length_limit_and_zero_budget():
    model = ScriptedModel([[2]])
    prompt = torch.tensor([[0]])
    assert generate(model, prompt, 0, eos_token_id=4) is prompt
    assert model.calls == 0
    assert generate(model, prompt, 2, temperature=0, eos_token_id=4).tolist() == [[0, 2, 2]]


@pytest.mark.parametrize("eos", [-1, 5, True, 4.0])
def test_invalid_eos_rejected(eos):
    with pytest.raises(ValueError, match="eos_token_id"):
        generate(ScriptedModel([[4]]), torch.tensor([[0]]), 1, eos_token_id=eos)
