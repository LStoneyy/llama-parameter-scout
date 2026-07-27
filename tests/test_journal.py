from __future__ import annotations

import json
from pathlib import Path

from llama_parameter_scout.journal import write_observation
from llama_parameter_scout.models import (
    BenchmarkObservation,
    CandidateConfig,
    ModelFingerprint,
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
        assert data["version"] == 2
        assert data["candidate"]["threads"] == 4
        assert data["total_ctx"] == 2048
        assert data["prompt_ctx"] == 512
        assert data["generation_ctx"] == 1024
        assert data["depth_ctx"] == 512
        assert data["samples_tps"] == [45.5, 44.8, 45.3]
        assert data["median_tps"] == 45.3
        assert isinstance(data["dispersion_tps"], float)
        assert data["n_measurements"] == 3
        assert data["binary_path"] == "/usr/local/bin/llama-bench"
        assert data["binary_source"] == "explicit"
        assert data["model_path"] == "/tmp/model.gguf"

    def test_toolchain_in_journal(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        data = json.loads(path.read_text())
        tc = data["toolchain"]
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
        data = json.loads(path.read_text())
        m = data["model"]
        assert m["architecture"] == "llama"
        assert m["parameter_count"] == 7_000_000_000
        assert m["context_length"] == 4096
        assert m["quantization"] == "Q4_K_M"
        assert m["shards"] == []
        assert m["symlink_warning"] is False
        assert m["network_mount_detected"] is False
        assert m["diagnostics"] == []

    def test_shards_in_journal(self, tmp_path: Path) -> None:
        from llama_parameter_scout.models import ShardFingerprint
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
        data = json.loads(path.read_text())
        shards = data["model"]["shards"]
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
