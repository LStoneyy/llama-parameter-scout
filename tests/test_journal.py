from __future__ import annotations

import json
from pathlib import Path

from llama_parameter_scout.journal import write_observation
from llama_parameter_scout.models import BenchmarkObservation, CandidateConfig


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
        assert data["version"] == 1
        assert data["candidate"]["threads"] == 4
        assert data["total_ctx"] == 2048
        assert data["prompt_ctx"] == 512
        assert data["generation_ctx"] == 1024
        assert data["depth_ctx"] == 512
        assert data["samples_tps"] == [45.5, 44.8, 45.3]
        assert data["median_tps"] == 45.3
        assert isinstance(data["dispersion_tps"], float)
        assert data["n_measurements"] == 3

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        assert path.exists()

    def test_atomic_write_no_partial(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        # Confirm there's no temp file left
        temps = list(tmp_path.glob(".*.json.*.tmp"))
        assert len(temps) == 0
