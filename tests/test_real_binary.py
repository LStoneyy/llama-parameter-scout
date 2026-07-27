from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llama_parameter_scout.application import Application
from llama_parameter_scout.models import CandidateConfig, NctxComposition, SingleRunRequest
from llama_parameter_scout.runner import ProductionRunner

pytestmark = pytest.mark.real_binary

LLAMA_BENCH_PATH = os.environ.get("LLAMA_BENCH_PATH") or ""
MODEL_PATH = os.environ.get("MODEL_PATH") or ""


@pytest.mark.skipif(
    not LLAMA_BENCH_PATH or not MODEL_PATH,
    reason="Set LLAMA_BENCH_PATH and MODEL_PATH env vars to run real-binary smoke test",
)
def test_real_binary_smoke(tmp_path: Path) -> None:
    request = SingleRunRequest(
        binary_path=Path(LLAMA_BENCH_PATH),
        model_path=Path(MODEL_PATH),
        journal_path=tmp_path / "journal.json",
        n_ctx=NctxComposition(total=512, prompt=256, generation=256, depth=0),
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
    runner = ProductionRunner()
    app = Application(runner)
    obs = app.run(request)

    assert len(obs.samples_tps) >= 3
    assert all(s > 0 for s in obs.samples_tps)
    assert obs.median_tps > 0
    assert obs.dispersion_tps >= 0
    assert obs.candidate.threads == 4

    assert request.journal_path.exists()
    data = json.loads(request.journal_path.read_text())
    assert data["version"] == 2
    assert data["n_measurements"] >= 3
