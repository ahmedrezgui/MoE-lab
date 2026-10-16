"""BPE vocabulary completeness, bounded samples, and legacy file compatibility."""

import json

import pytest

from moelab.tokenizer.bpe import BPETokenizer


def test_requested_capacity_can_exceed_actual_vocab_and_retraining(tmp_path):
    tokenizer = BPETokenizer(512)
    tokenizer.train("abab")
    assert tokenizer.requested_vocab_size == 512
    assert tokenizer.vocab_size == 257
    assert tokenizer.encode("abab") == [256, 256]
    # Every class the model would predict has a defined byte expansion.
    for token in range(tokenizer.vocab_size):
        assert isinstance(tokenizer.decode([token]), str)
    with pytest.raises(ValueError, match="actual vocabulary"):
        tokenizer.decode([257])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.vocab_size == 257
    assert restored.requested_vocab_size == 512
    restored.train("xyz")
    assert restored.vocab_size == 256 and not restored.merges


def test_document_boundaries_are_not_merge_candidates():
    tokenizer = BPETokenizer(300)
    tokenizer.train_documents(["a", "b", "a", "b"])
    assert tokenizer.vocab_size == 256
    tokenizer.train_documents(["ab", "ab"])
    assert tokenizer.merges == {(97, 98): 256}


def test_sample_byte_limit_and_utf8_fragment():
    def texts():
        yield "caf\u00e9 caf\u00e9"
        pytest.fail("Bound exhausted: later document should not be read")

    tokenizer = BPETokenizer(300)
    stats = tokenizer.train_documents(texts(), max_training_bytes=4)
    assert stats["bytes"] == 4
    assert stats["documents"] == 1
    assert stats["last_document_truncated"]
    assert tokenizer.decode(tokenizer.encode("caf\u00e9 \U0001f98a")) == "caf\u00e9 \U0001f98a"
    with pytest.raises(ValueError, match="positive"):
        tokenizer.train("text", max_training_bytes=0)


def test_legacy_requested_size_is_recovered_and_new_metadata_is_validated(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"vocab_size": 512, "merges": [[97, 98, 256]]}))
    tokenizer = BPETokenizer.load(path)
    assert tokenizer.vocab_size == 257
    assert tokenizer.requested_vocab_size == 512
    assert tokenizer.decode([256]) == "ab"
    tokenizer.save(path)
    payload = json.loads(path.read_text())
    assert payload["vocab_size"] == 257
    payload["vocab_size"] = 512
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        BPETokenizer.load(path)


def test_fitting_and_sample_hash_are_deterministic():
    a, b = BPETokenizer(300), BPETokenizer(300)
    texts = ["abracadabra", "banana banana"]
    assert a.train_documents(texts, 16) == b.train_documents(iter(texts), 16)
    assert a.merges == b.merges


def test_eos_is_reserved_and_explicit(tmp_path):
    with pytest.raises(ValueError, match="optional EOS"):
        BPETokenizer(256, add_eos=True)
    byte = BPETokenizer(257, add_eos=True)
    byte.train("abab")
    assert not byte.merges and byte.eos_token_id == 256
    tokenizer = BPETokenizer(512, add_eos=True)
    tokenizer.train("abab")
    assert tokenizer.vocab_size == 258 and tokenizer.eos_token_id == 257
    assert tokenizer.encode("abab", add_eos=True) == [256, 256, 257]
    literal = "<|endoftext|>"
    assert 257 not in tokenizer.encode(literal)
    assert tokenizer.decode(tokenizer.encode(literal)) == literal
    assert tokenizer.decode([256, 257]) == "ab"
    assert tokenizer.decode([256, 257], skip_special_tokens=False) == "ab" + literal
    with pytest.raises(ValueError, match="no EOS"):
        BPETokenizer().encode("text", add_eos=True)
    path = tmp_path / "eos.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.encode("abab", add_eos=True) == [256, 256, 257]
    payload = json.loads(path.read_text())
    payload["eos_token_id"] = 256
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Stored EOS"):
        BPETokenizer.load(path)


def test_format_two_remains_without_eos(tmp_path):
    path = tmp_path / "v2.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "requested_vocab_size": 512,
                "vocab_size": 257,
                "merges": [[97, 98, 256]],
            }
        )
    )
    tokenizer = BPETokenizer.load(path)
    assert tokenizer.eos_token_id is None
    assert tokenizer.decode([256]) == "ab"
