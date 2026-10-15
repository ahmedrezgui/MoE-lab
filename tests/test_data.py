import numpy as np
import pytest

from moelab.data.dataset import BlockDataset
from moelab.tokenizer.bpe import BPETokenizer


def test_bpe_unicode_roundtrip_and_serialization(tmp_path):
    text = "hello hello hello! Bonjour, café. 你好 U0001f98a"
    tokenizer = BPETokenizer(280)
    tokenizer.train(text)
    assert tokenizer.merges
    assert len(tokenizer.encode(text)) < len(text.encode())
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    for sample in [text, "", "unseen ΩU0001f642"]:
        assert restored.decode(restored.encode(sample)) == sample
        assert restored.encode(sample) == tokenizer.encode(sample)


def test_dataset_includes_last_valid_block(tmp_path):
    path = tmp_path / "ids.bin"
    np.arange(5, dtype="<u2").tofile(path)
    dataset = BlockDataset(path, 4)
    assert len(dataset) == 1
    x, y = dataset.get_batch(3, np.random.default_rng(42))
    np.testing.assert_array_equal(x, [[0, 1, 2, 3]] * 3)
    np.testing.assert_array_equal(y, x + 1)
    with pytest.raises(ValueError):
        BlockDataset(path, 5)
