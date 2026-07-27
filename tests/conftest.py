from __future__ import annotations

import json
from pathlib import Path

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    NctxComposition,
    SingleRunRequest,
)
from llama_parameter_scout.runner import FakeRunner


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    p = tmp_path / "llama-bench"
    p.write_text("#!/bin/sh\necho hello")
    p.chmod(0o755)
    return p


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    p = tmp_path / "model.gguf"
    p.write_text("fake gguf data")
    return p


@pytest.fixture
def valid_request(
    fake_binary: Path,
    fake_model: Path,
    tmp_path: Path,
) -> SingleRunRequest:
    return SingleRunRequest(
        binary_path=fake_binary,
        model_path=fake_model,
        journal_path=tmp_path / "journal.json",
        n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
        candidate=CandidateConfig(
            threads=4,
            batch_size=512,
            ubatch_size=512,
            n_gpu_layers=0,
            flash_attn=False,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
    )


@pytest.fixture
def valid_json_output() -> str:
    records = [
        {
            "n_ctx": 2048,
            "n_batch": 512,
            "n_ubatch": 512,
            "n_threads": 4,
            "n_gpu_layers": 0,
            "main_gpu": 0,
            "flash_attn": False,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "test_time": 42.5,
            "avg_ts": 45.2,
            "stddev_ts": 0.5,
            "samples_ts": [45.5, 44.8, 45.3],
        },
    ]
    return json.dumps(records)


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()
