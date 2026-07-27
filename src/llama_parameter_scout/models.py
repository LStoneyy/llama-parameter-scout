from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ── Error hierarchy ──────────────────────────────────────────────


class ScoutError(Exception):
    """Base error for all scout-domain errors."""


class BinaryNotFoundError(ScoutError):
    """The llama-bench binary does not exist or is not executable."""


class ModelNotFoundError(ScoutError):
    """The GGUF model file does not exist."""


class InvalidNctxError(ScoutError):
    """The n_ctx composition is invalid."""


class CandidateError(ScoutError):
    """A candidate configuration value is out of range."""


class SubprocessError(ScoutError):
    """The llama-bench subprocess exited with a non-zero code."""


class MalformedOutputError(ScoutError):
    """llama-bench output could not be parsed as valid machine-readable JSON."""


class JournalError(ScoutError):
    """Failed to atomically write the journal file."""


class ToolchainError(ScoutError):
    """Toolchain resolution or inspection failed."""


class ModelRejectedError(ScoutError):
    """The model path or format is not acceptable."""


# ── Domain models ────────────────────────────────────────────────

BinarySource = Literal["explicit", "install_dir", "path"]


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
    binary_source: BinarySource = "explicit"


@dataclass(frozen=True)
class BinaryResolution:
    binary_path: Path
    source: BinarySource


@dataclass(frozen=True)
class ToolchainFingerprint:
    binary_path: Path
    version: str
    supported_flags: tuple[str, ...]
    supported_output_formats: tuple[str, ...]
    device_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ShardFingerprint:
    path: Path
    size_bytes: int
    modified_time: float
    digest_sha256: str


@dataclass(frozen=True)
class ModelFingerprint:
    path: Path
    architecture: str
    parameter_count: int | None
    context_length: int | None
    quantization: str | None
    shards: tuple[ShardFingerprint, ...]
    symlink_warning: bool
    network_mount_detected: bool
    diagnostics: tuple[str, ...]


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
    binary_path: Path
    binary_source: BinarySource
    model_path: Path
    toolchain: ToolchainFingerprint
    model: ModelFingerprint
