import json

import numpy as np
import pytest

from moelab.data.confirmation import prepare_confirmation
from moelab.data.prepare import prepare_documents


def records(path, texts):
    path.write_text("\n".join(json.dumps({"text": x}) for x in texts), encoding="utf-8")
    return path


@pytest.fixture
def fixture(tmp_path):
    train = records(tmp_path / "train.jsonl", ["training story", "other train"])
    val = records(tmp_path / "val.jsonl", ["development story"])
    prepared = tmp_path / "prepared"
    prepare_documents(train, val, prepared, block_size=1, vocab_size=257, add_eos=True)
    source = records(
        tmp_path / "source.jsonl",
        [
            "development story",
            " training story ",
            "development story",
            "fresh\r\nstory",
            "fresh\nstory",
            "second fresh story",
        ],
    )
    return dict(
        source=source,
        prior_train=train,
        prior_val=val,
        training_data_dir=prepared,
        output=tmp_path / "confirmation",
        start_document=1,
        document_count=5,
    )


def test_fixed_slice_excludes_overlaps_and_freezes_tokenizer(fixture):
    result = prepare_confirmation(**fixture)
    assert result["counts"] == {
        "requested": 5,
        "kept": 2,
        "train_overlap": 1,
        "development_overlap": 1,
        "duplicates_within_slice": 1,
    }
    output = fixture["output"]
    assert (output / "tokenizer.json").read_bytes() == (
        fixture["training_data_dir"] / "tokenizer.json"
    ).read_bytes()
    from moelab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(output / "tokenizer.json")
    ids = np.fromfile(output / "ids_confirmation.bin", dtype="<u2").tolist()
    expected = tokenizer.encode("fresh\nstory", add_eos=True) + tokenizer.encode(
        "second fresh story", add_eos=True
    )
    assert ids == expected
    assert ids.count(tokenizer.eos_token_id) == 2
    assert len(result["retained_document_sha256"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        prepare_confirmation(**fixture)


@pytest.mark.parametrize("change", ["prefix", "short", "raw", "tokenizer", "empty"])
def test_bad_provenance_or_selection_leaves_no_output(fixture, change):
    if change == "prefix":
        records(fixture["source"], ["different"] * 6)
    elif change == "short":
        fixture["document_count"] = 6
    elif change == "raw":
        fixture["prior_train"].write_text('{"text":"changed"}')
    elif change == "tokenizer":
        with (fixture["training_data_dir"] / "tokenizer.json").open("a") as stream:
            stream.write(" ")
    else:
        records(fixture["source"], ["development story"] * 6)
    with pytest.raises(ValueError):
        prepare_confirmation(**fixture)
    assert not fixture["output"].exists()


def test_confirmation_checks_checkpoints_and_original_data(fixture, tmp_path):
    import torch

    from moelab.comparison import fingerprint
    from moelab.config import ModelConfig
    from moelab.evals.confirmation import evaluate_confirmation
    from moelab.model.transformer import Transformer

    torch.set_num_threads(2)
    metadata = prepare_confirmation(**fixture)
    root = tmp_path / "ablation"
    root.mkdir()
    runs = []
    for coef in (0.01, 0.1):
        cfg = ModelConfig(
            vocab_size=257,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            moe={"n_experts": 2, "top_k": 1, "load_balance_coef": coef},
        )
        name = f"coef-{coef}-seed-42"
        directory = root / name
        directory.mkdir()
        checkpoint = directory / "last.pt"
        training = {"seed": 42}
        torch.save(
            {
                "model": Transformer(cfg).state_dict(),
                "model_config": cfg.model_dump(),
                "training_config": training,
                "step": 2,
                "data_hashes": metadata["training_data_hashes"],
            },
            checkpoint,
        )
        runs.append(
            {
                "run_name": name,
                "coefficient": coef,
                "seed": 42,
                "training_summary": {"model_config": cfg.model_dump(), "training_config": training},
                "full_validation": {"val_ce": 5.0, "checkpoint_sha256": fingerprint(checkpoint)},
            }
        )
    manifest = {
        "status": "complete",
        "steps": 2,
        "data_hashes": metadata["training_data_hashes"],
        "seeds": [42],
        "runs": runs,
    }
    (root / "ablation.json").write_text(json.dumps(manifest))
    output = tmp_path / "evaluation"
    result = evaluate_confirmation(root, fixture["output"], output, device="cpu")
    assert result["status"] == "complete" and len(result["runs"]) == 2
    assert all(r["scored_targets"] == metadata["tokens"] - 1 for r in result["runs"])
    assert result["mean_paired_difference"] == pytest.approx(
        result["runs"][1]["val_ce"] - result["runs"][0]["val_ce"]
    )
    with pytest.raises(FileExistsError):
        evaluate_confirmation(root, fixture["output"], output, device="cpu")
    # Changes to the saved checkpoint must fail before scoring the changed weights.
    with checkpoint.open("ab") as stream:
        stream.write(b"changed")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="Checkpoint differs"):
        evaluate_confirmation(root, fixture["output"], failed, device="cpu")
    assert json.loads((failed / "confirmation.json").read_text())["status"] == "failed"
    with (fixture["output"] / "ids_confirmation.bin").open("ab") as stream:
        stream.write(b"xx")
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_confirmation(root, fixture["output"], tmp_path / "bad-data", device="cpu")
