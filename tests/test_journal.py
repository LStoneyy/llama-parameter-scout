from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, cast

import pytest

from llama_parameter_scout.journal import (
    _read_journal,
    read_journal_events,
    write_observation,
)
from llama_parameter_scout.models import (
    BenchmarkObservation,
    CandidateConfig,
    JournalError,
    ModelFingerprint,
    ShardFingerprint,
    ToolchainFingerprint,
)


def _make_observation() -> BenchmarkObservation:
    return BenchmarkObservation(
        candidate=CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512, n_gpu_layers=0,
            flash_attn=False, cache_type_k="f16", cache_type_v="f16",
        ),
        total_ctx=2048,
        prompt_ctx=512,
        generation_ctx=1024,
        depth_ctx=512,
        samples_tps=(45.5, 44.8, 45.3),
        median_tps=45.3,
        dispersion_tps=0.286,
        binary_path=Path("/usr/local/bin/llama-bench"),
        binary_source="explicit",
        model_path=Path("/tmp/model.gguf"),
        toolchain=ToolchainFingerprint(
            binary_path=Path("/usr/local/bin/llama-bench"),
            version="1234",
            supported_flags=(
                "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
            ),
            supported_output_formats=("json",),
            device_capabilities=("cpu", "metal"),
        ),
        model=ModelFingerprint(
            path=Path("/tmp/model.gguf"),
            architecture="llama",
            parameter_count=7_000_000_000,
            context_length=4096,
            quantization="Q4_K_M",
            shards=(),
            symlink_warning=False,
            network_mount_detected=False,
            diagnostics=(),
        ),
    )


def _obs_data(path: Path) -> dict[str, Any]:
    events = read_journal_events(path)
    assert len(events) >= 1
    ev = events[0]
    assert ev["event_type"] == "observation_completed"
    obs = ev["observation"]
    assert isinstance(obs, dict)
    return obs


class TestWriteObservation:
    def test_writes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        assert path.exists()

    def test_content_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        data = json.loads(path.read_text())
        assert data["schema_version"] == 5
        assert len(data["events"]) == 1
        ev = data["events"][0]
        assert ev["event_type"] == "observation_completed"
        assert ev["event_version"] == 1
        assert ev["session_id"] != ""
        o = ev["observation"]
        assert o["version"] == 4
        assert o["candidate"]["threads"] == 4
        assert o["total_ctx"] == 2048
        assert o["prompt_ctx"] == 512
        assert o["generation_ctx"] == 1024
        assert o["depth_ctx"] == 512
        assert o["samples_tps"] == [45.5, 44.8, 45.3]
        assert o["median_tps"] == 45.3
        assert isinstance(o["dispersion_tps"], float)
        assert o["n_measurements"] == 3
        assert o["binary_path"] == "/usr/local/bin/llama-bench"
        assert o["binary_source"] == "explicit"
        assert o["model_path"] == "/tmp/model.gguf"

    def test_toolchain_in_journal(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        o = _obs_data(path)
        tc = o["toolchain"]
        assert tc["binary_path"] == "/usr/local/bin/llama-bench"
        assert tc["version"] == "1234"
        assert tc["supported_flags"] == [
            "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
            "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
        ]
        assert tc["supported_output_formats"] == ["json"]
        assert tc["device_capabilities"] == ["cpu", "metal"]

    def test_model_in_journal(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        o = _obs_data(path)
        m = o["model"]
        assert m["architecture"] == "llama"
        assert m["parameter_count"] == 7_000_000_000
        assert m["context_length"] == 4096
        assert m["quantization"] == "Q4_K_M"
        assert m["shards"] == []
        assert m["symlink_warning"] is False
        assert m["network_mount_detected"] is False
        assert m["diagnostics"] == []

    def test_shards_in_journal(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        obs = BenchmarkObservation(
            candidate=obs.candidate,
            total_ctx=obs.total_ctx,
            prompt_ctx=obs.prompt_ctx,
            generation_ctx=obs.generation_ctx,
            depth_ctx=obs.depth_ctx,
            samples_tps=obs.samples_tps,
            median_tps=obs.median_tps,
            dispersion_tps=obs.dispersion_tps,
            binary_path=obs.binary_path,
            binary_source=obs.binary_source,
            model_path=obs.model_path,
            toolchain=obs.toolchain,
            model=ModelFingerprint(
                path=obs.model.path,
                architecture=obs.model.architecture,
                parameter_count=obs.model.parameter_count,
                context_length=obs.model.context_length,
                quantization=obs.model.quantization,
                shards=(
                    ShardFingerprint(
                        path=Path("/tmp/model-00001-of-00002.gguf"),
                        size_bytes=12345,
                        modified_time=1000000.0,
                        digest_sha256="a" * 64,
                    ),
                    ShardFingerprint(
                        path=Path("/tmp/model-00002-of-00002.gguf"),
                        size_bytes=67890,
                        modified_time=2000000.0,
                        digest_sha256="b" * 64,
                    ),
                ),
                symlink_warning=obs.model.symlink_warning,
                network_mount_detected=obs.model.network_mount_detected,
                diagnostics=obs.model.diagnostics,
            ),
        )
        write_observation(path, obs)
        o = _obs_data(path)
        shards = o["model"]["shards"]
        assert len(shards) == 2
        assert shards[0]["path"] == "/tmp/model-00001-of-00002.gguf"
        assert shards[0]["size_bytes"] == 12345
        assert shards[0]["modified_time"] == 1000000.0
        assert shards[0]["digest_sha256"] == "a" * 64

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        assert path.exists()

    def test_atomic_write_no_partial(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        temps = list(tmp_path.glob(".*.json.*.tmp"))
        assert len(temps) == 0


class TestReadJournalSecurity:
    def test_empty_file_raises_journal_error(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_text("")
        with pytest.raises(JournalError, match="empty"):
            _read_journal(path)

    def test_invalid_utf8_raises_journal_error(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(JournalError, match="UTF-8"):
            _read_journal(path)

    def test_journal_at_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        small_limit = 64
        monkeypatch.setattr("llama_parameter_scout.journal._MAX_JOURNAL_BYTES", small_limit)
        path = tmp_path / "journal.json"
        body = '{"schema_version": 5, "events": []}'
        content = body + " " * (small_limit - len(body))
        assert len(content) == small_limit
        path.write_text(content)
        result = _read_journal(path)
        assert result is not None
        assert result["schema_version"] == 5
        assert result["events"] == []

    def test_journal_exceeds_limit_by_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        small_limit = 64
        monkeypatch.setattr("llama_parameter_scout.journal._MAX_JOURNAL_BYTES", small_limit)
        path = tmp_path / "journal.json"
        path.write_text("x" * (small_limit + 1))
        with pytest.raises(JournalError, match="exceeds"):
            _read_journal(path)

    def test_sparse_file_bounded_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        small_limit = 64
        monkeypatch.setattr("llama_parameter_scout.journal._MAX_JOURNAL_BYTES", small_limit)
        path = tmp_path / "journal.json"
        with open(path, "wb") as f:
            f.write(b"{}")
            f.seek(small_limit * 100)
            f.write(b" ")

        read_args: list[int] = []

        def spy_read(orig_read: Any, n: int = -1) -> bytes:
            read_args.append(n)
            return cast(bytes, orig_read(n))

        original_open = builtins.open

        class _SpyFile:
            def __init__(self, inner: Any) -> None:
                self._inner = inner
            def __enter__(self) -> _SpyFile:
                return self
            def __exit__(self, *args: object) -> None:
                self._inner.close()
            def read(self, n: int = -1) -> bytes:
                return spy_read(self._inner.read, n)

        def _spy_open(path: Any, mode: str = "rb") -> _SpyFile:
            return _SpyFile(original_open(path, mode))

        monkeypatch.setattr(builtins, "open", _spy_open)

        with pytest.raises(JournalError):
            _read_journal(path)

        assert read_args == [small_limit + 1]
