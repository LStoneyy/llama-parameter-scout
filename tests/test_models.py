from __future__ import annotations

from llama_parameter_scout.models import CandidateConfig, NctxComposition, SingleRunRequest


def test_candidate_config_fields() -> None:
    c = CandidateConfig(
        threads=4,
        batch_size=512,
        ubatch_size=512,
        n_gpu_layers=0,
        flash_attn=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    assert c.threads == 4
    assert c.cache_type_k == "f16"


def test_nctx_composition_fields() -> None:
    n = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
    assert n.total == 2048
    assert n.prompt + n.generation + n.depth == n.total


def test_request_fields(valid_request: SingleRunRequest) -> None:
    assert valid_request.n_ctx.total == 2048
    assert valid_request.candidate.threads == 4
    assert valid_request.journal_path.name == "journal.json"
