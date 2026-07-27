from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateConfig:
    threads: int
    batch_size: int
    ubatch_size: int
    n_gpu_layers: int
    flash_attn: bool
    cache_type_k: str
    cache_type_v: str


@dataclass(frozen=True)
class NctxComposition:
    total: int
    prompt: int
    generation: int
    depth: int


@dataclass(frozen=True)
class SingleRunRequest:
    binary_path: Path
    model_path: Path
    journal_path: Path
    n_ctx: NctxComposition
    candidate: CandidateConfig


@dataclass(frozen=True)
class BenchmarkObservation:
    candidate: CandidateConfig
    total_ctx: int
    prompt_ctx: int
    generation_ctx: int
    depth_ctx: int
    samples_tps: tuple[float, ...]
    median_tps: float
    dispersion_tps: float
