import io
import json
from urllib.error import URLError

import pytest

from moelab.data import download
from moelab.data.prepare import prepare_documents

CARD = b"---\nlicense: cdla-sharing-1.0\n---\nDataset attribution.\n"


class ChunkedSource(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))


def install_sources(monkeypatch, train=None, val=None, card=CARD):
    if train is None:
        train = "Training caf\u00e9 story.<|endoftext|>Second training story.<|endoftext|>".encode()
    if val is None:
        val = b"Validation story.<|endoftext|>"
    requests = []

    def open_source(url, limit):
        requests.append((url, limit))
        if url.endswith("README.md"):
            return io.BytesIO(card)
        payload = train if url.endswith("train.txt") else val
        if isinstance(payload, Exception):
            raise payload
        return ChunkedSource(payload)

    monkeypatch.setattr(download, "open_source", open_source)
    return requests


def test_pinned_download_reproducible_and_preparable(tmp_path, monkeypatch):
    requests = install_sources(monkeypatch)
    output = tmp_path / "raw"
    manifest = download.download_tinystories(output, 2, 1)
    assert all(download.REVISION in url for url, _ in requests)
    assert manifest["splits"]["train"]["documents"] == 2
    assert manifest["license"] == "cdla-sharing-1.0"
    assert (output / "SOURCE_README.md").read_bytes() == CARD
    records = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["text"] == "Training caf\u00e9 story."
    assert "<|endoftext|>" not in records[0]["text"]
    second = tmp_path / "repeat"
    assert download.download_tinystories(second, 2, 1) == manifest
    for file in output.iterdir():
        assert file.read_bytes() == (second / file.name).read_bytes()
    prepared = prepare_documents(
        output / "train.jsonl", output / "val.jsonl", tmp_path / "prepared", block_size=4
    )
    assert prepared["documents"]["train"]["kept"] == 2
    assert prepared["documents"]["val"]["kept"] == 1


@pytest.mark.parametrize(
    "payload,kwargs,match",
    [
        (b"unfinished", {}, "complete documents"),
        (b"story<|endoftext|>", {"max_bytes_per_split": 5}, "complete documents"),
        (b"a" * 40 + b"<|endoftext|>", {"max_document_bytes": 10}, "max_document_bytes"),
        (b"\xff<|endoftext|>", {}, "utf-8"),
    ],
)
def test_invalid_or_limited_source_leaves_no_output(tmp_path, monkeypatch, payload, kwargs, match):
    install_sources(monkeypatch, train=payload)
    with pytest.raises(ValueError, match=match):
        download.download_tinystories(tmp_path / "raw", 1, 1, **kwargs)
    assert not (tmp_path / "raw").exists()
    assert not list(tmp_path.glob(".download-*"))


def test_network_failure_preserves_existing_files(tmp_path, monkeypatch):
    install_sources(monkeypatch, val=URLError("offline"))
    with pytest.raises(URLError):
        download.download_tinystories(tmp_path / "raw", 1, 1)
    assert not (tmp_path / "raw").exists()
    assert not list(tmp_path.glob(".download-*"))
    output = tmp_path / "raw"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        download.download_tinystories(output, 1, 1)
    assert (output / "keep").read_text() == "keep"


@pytest.mark.parametrize(
    "card",
    [b"missing front matter", b"---\nlicense: other\n---\n", b"x" * 65537],
    ids=["missing", "changed-license", "oversized"],
)
def test_rejects_unknown_license_or_oversized_card(tmp_path, monkeypatch, card):
    install_sources(monkeypatch, card=card)
    with pytest.raises(ValueError):
        download.download_tinystories(tmp_path / "raw", 1, 1)
    assert not (tmp_path / "raw").exists()


def test_arguments_fail_before_network(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("Invalid arguments must not make network requests")

    monkeypatch.setattr(download, "open_source", unexpected)
    for kwargs in [{"revision": "main"}, {"train_documents": 0}, {"max_bytes_per_split": 0}]:
        with pytest.raises(ValueError):
            download.download_tinystories(tmp_path / "raw", **kwargs)


def test_source_read_budget_is_respected(tmp_path, monkeypatch):
    class CountedSource(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            result = super().read(size)
            self.consumed += len(result)
            return result

    source = CountedSource(b"a" * 1000)
    monkeypatch.setattr(download, "open_source", lambda *a: source)
    with pytest.raises(ValueError, match="complete documents"):
        download.export_split("https://example.test/source", tmp_path / "out.jsonl", 1, 17, 100)
    assert source.consumed == 17
