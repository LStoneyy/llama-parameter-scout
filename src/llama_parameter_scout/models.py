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


class SessionSetupError(ScoutError):
    """Base for session-specification errors."""


class NoAcceleratorSelectedError(SessionSetupError):
    """Acceleration was chosen but no accelerator device was selected."""


class MultipleAcceleratorsSelectedError(SessionSetupError):
    """More than one accelerator device ID was provided."""


class UnknownDeviceError(SessionSetupError):
    """The selected device ID does not match any detected accelerator."""


class InconsistentNglError(SessionSetupError):
    """n_gpu_layers is inconsistent with CPU-only / device selection."""


class ModelFingerprintMismatchError(SessionSetupError):
    """Session spec model fingerprint does not match inspected model."""


# ── Session domain models ────────────────────────────────────────


@dataclass(frozen=True)
class Accelerator:
    kind: str
    backend: str
    device_id: int
    description: str
    memory_bytes: int | None = None


@dataclass(frozen=True)
class SystemTopology:
    cpu_count: int
    system_memory_bytes: int
    accelerators: tuple[Accelerator, ...]
    memory_unified: bool


@dataclass(frozen=True)
class ResourceBudget:
    ram_bytes: int
    vram_bytes: int | None
    headroom_ratio: float


class ContextExceededWithoutAcknowledgementError(SessionSetupError):
    """n_ctx exceeds model training context and was not acknowledged."""


class NonpositiveBudgetError(SessionSetupError):
    """Resource budget values must be positive."""


class InvalidHeadroomRatioError(SessionSetupError):
    """Headroom ratio must be in [0.0, 1.0)."""


TrainingContextStatus = Literal["known", "unknown", "conflicted"]


@dataclass(frozen=True)
class ContextAdvisory:
    likely_fit_lower: int
    likely_fit_upper: int
    uncertain_maximum: int
    model_weight_bytes: int
    kv_cache_bytes_estimate: int
    training_context_status: TrainingContextStatus = "unknown"
    training_context_tokens: int | None = None
    note: str = ""


@dataclass(frozen=True)
class SessionSpecification:
    topology: SystemTopology
    cpu_only: bool
    selected_device_ids: tuple[int, ...]
    n_ctx: NctxComposition
    budget: ResourceBudget
    advisory: ContextAdvisory
    model: ModelFingerprint
    acknowledged: bool
    fingerprint: str


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
    session_spec: SessionSpecification | None = None
