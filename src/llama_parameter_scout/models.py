from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Literal

ShellDialect = Literal["posix", "powershell", "cmd"]

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


class IncompatibleDraftError(ScoutError):
    """Main model and draft model are not compatible for MTP."""


class ServerCapabilityError(ScoutError):
    """llama-server does not support required capabilities."""


class MtpMeasurementError(ServerCapabilityError):
    """Server completion response was malformed or invalid."""


# ── Execution domain models ─────────────────────────────────────


CandidateDisposition = Literal[
    "unsupported",
    "pruned",
    "not_attempted",
    "budget_exhausted",
    "oom",
    "timeout",
    "crash",
    "malformed_output",
    "cancellation",
    "systemic_error",
    "completed",
]


@dataclass(frozen=True)
class ExecutionPolicy:
    env_overrides: dict[str, str] | None = None
    candidate_timeout_seconds: float | None = None
    cancellation: Event | None = None
    memory_pressure_fn: Callable[[], bool] | None = None
    memory_pressure_backoff_seconds: float = 1.0
    memory_pressure_max_backoffs: int = 3

    def __post_init__(self) -> None:
        if self.candidate_timeout_seconds is not None and self.candidate_timeout_seconds <= 0:
            raise ValueError("candidate_timeout_seconds must be positive if provided")
        if self.memory_pressure_backoff_seconds < 0:
            raise ValueError("memory_pressure_backoff_seconds must be nonnegative")
        if self.memory_pressure_max_backoffs <= 0:
            raise ValueError("memory_pressure_max_backoffs must be positive")


@dataclass(frozen=True)
class CandidateOutcome:
    candidate: CandidateConfig
    disposition: CandidateDisposition
    process_launched: bool
    diagnostic: str
    safe_next_action: str
    observation: BenchmarkObservation | None = None


@dataclass
class StopTracker:
    max_stops: int = 3
    _consecutive: int = field(default=0, repr=False)

    def record(self, outcome: CandidateOutcome) -> bool:
        if outcome.disposition in ("systemic_error", "crash", "oom"):
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self.should_stop

    def reset(self) -> None:
        self._consecutive = 0

    @property
    def should_stop(self) -> bool:
        return self._consecutive >= self.max_stops


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
    n_cpu_moe: int | None = None


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
    supported_cache_types: tuple[str, ...] = ("f16",)


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
    is_moe: bool = False
    moe_expert_count: int | None = None


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


# ── Memory evidence model ─────────────────────────────────────────


MemoryQuality = Literal["complete", "partial", "estimated", "incomparable"]


@dataclass(frozen=True)
class MemoryEvidence:
    ram_bytes: int | None = None
    vram_bytes: int | None = None
    quality: MemoryQuality | None = None


# ── Pareto frontier entry ─────────────────────────────────────────


@dataclass(frozen=True)
class ParetoEntry:
    candidate: CandidateConfig
    balanced: float
    memory: MemoryEvidence


# ── Search / Screening domain models ──────────────────────────────


ScreeningDisposition = Literal[
    "survived",
    "pruned",
    "dominance_pruned",
    "budget_exhausted",
    "oom",
    "timeout",
    "crash",
    "malformed_output",
    "systemic_error",
    "unsupported",
    "cancellation",
    "not_attempted",
    "matches_baseline",
]


@dataclass(frozen=True)
class SearchProfile:
    label: str
    max_attempts: int
    min_repetitions: int
    max_repetitions_per_candidate: int
    tie_bound_coefficient: float
    dispersion_threshold: float

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.min_repetitions < 3:
            raise ValueError("min_repetitions must be >= 3 (user story 31)")
        if self.max_repetitions_per_candidate < self.min_repetitions:
            raise ValueError(
                "max_repetitions_per_candidate must be >= min_repetitions"
            )
        if not 0.0 <= self.tie_bound_coefficient <= 1.0:
            raise ValueError("tie_bound_coefficient must be in [0.0, 1.0]")
        if self.dispersion_threshold < 0:
            raise ValueError("dispersion_threshold must be >= 0")


@dataclass(frozen=True)
class ScreeningObservation:
    candidate: CandidateConfig
    pp_samples: tuple[float, ...]
    tg_samples: tuple[float, ...]
    memory_bytes: int | None = None
    memory: MemoryEvidence | None = None
    observed_prompt: int | None = None
    observed_generation: int | None = None
    observed_depth: int | None = None
    observed_total: int | None = None


@dataclass(frozen=True)
class ScreeningResult:
    candidate: CandidateConfig
    success: bool
    disposition: ScreeningDisposition = "survived"
    reason: str = ""
    observation: ScreeningObservation | None = None
    provenance: str = ""


@dataclass(frozen=True)
class RankedCandidate:
    candidate: CandidateConfig
    pp_score: float
    tg_score: float
    balanced: float
    pp_median: float
    tg_median: float
    pp_relative_dispersion: float
    tg_relative_dispersion: float
    memory_bytes: int | None = None
    memory_evidence: MemoryEvidence | None = None
    total_repetitions: int = 0
    tie_break: str = ""


@dataclass(frozen=True)
class ProvisionalLeader:
    candidate: CandidateConfig
    objective: str
    pp_score: float
    tg_score: float
    balanced: float
    pp_median: float
    tg_median: float
    label: str = "Screening evidence only - not validated at requested context"


@dataclass(frozen=True)
class ScreeningReport:
    profile: SearchProfile
    baseline_disposition: ScreeningResult
    grid_dispositions: tuple[ScreeningResult, ...]
    adaptive_dispositions: tuple[ScreeningResult, ...] = ()
    survivors: tuple[RankedCandidate, ...] = ()
    leader: ProvisionalLeader | None = None
    attempts_used: int = 0
    objective: str = "balanced"
    disposition_summary: str = ""
    notes: tuple[str, ...] = ()
    pareto_frontier: tuple[ParetoEntry, ...] = ()


# ── Validation domain models ──────────────────────────────────────


ValidationDisposition = Literal[
    "oom",
    "timeout",
    "crash",
    "malformed_output",
    "context_mismatch",
    "validation_budget_exhausted",
    "systemic_error",
]


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate: CandidateConfig
    pp_score: float
    tg_score: float
    balanced: float
    pp_median: float
    tg_median: float
    pp_relative_dispersion: float
    tg_relative_dispersion: float
    memory_evidence: MemoryEvidence | None = None
    label: str = "Full-context validation - verified at requested context"


@dataclass(frozen=True)
class FailedValidation:
    candidate: CandidateConfig
    disposition: str
    reason: str


@dataclass(frozen=True)
class ValidatedRecommendation:
    recommendation: ValidatedCandidate
    alternatives: tuple[ValidatedCandidate, ...]
    failures: tuple[FailedValidation, ...]
    attempts_used: int
    budget: int
    pareto_frontier: tuple[ParetoEntry, ...] = ()


@dataclass(frozen=True)
class NoWinnerReport:
    provisional_leader: ProvisionalLeader | None
    failures: tuple[FailedValidation, ...]
    attempts_used: int
    budget: int
    pareto_frontier: tuple[ParetoEntry, ...] = ()


# ── Journal v5 / Session domain ──────────────────────────────────


SessionResumeAction = Literal["new", "resumed", "derived"]


@dataclass(frozen=True)
class SessionResumeResult:
    action: SessionResumeAction
    session_id: str
    parent_session_id: str | None = None
    observation: BenchmarkObservation | None = None
    mismatch_reason: str | None = None
    completed_search_results: tuple[ScreeningResult, ...] = ()


# ── Export domain models ──────────────────────────────────────────


@dataclass(frozen=True)
class LaunchProvenance:
    cwd: str
    shell_dialect: ShellDialect
    executable_path: str
    argv: tuple[str, ...]
    environment: dict[str, str]

    def __post_init__(self) -> None:
        if self.shell_dialect not in ("posix", "powershell", "cmd"):
            raise ValueError(
                f"shell_dialect must be 'posix', 'powershell', or 'cmd', "
                f"got {self.shell_dialect!r}"
            )


# ── MTP domain models ─────────────────────────────────────────────


@dataclass(frozen=True)
class MtpMeasurement:
    mode: str
    samples_tps: tuple[float, ...]
    median_tps: float
    dispersion_tps: float


MtpBenefit = Literal["established", "not_established"]


@dataclass(frozen=True)
class MtpComparison:
    off: MtpMeasurement
    on: MtpMeasurement
    benefit: MtpBenefit
    improvement_pct: float | None
    tie_bound_tps: float
    caveat: str = ""


@dataclass(frozen=True)
class MtpEvidence:
    main_fingerprint: ModelFingerprint
    draft_fingerprint: ModelFingerprint
    server_path: str
    server_version: str
    server_supported_flags: tuple[str, ...]
    candidate: CandidateConfig
    n_ctx: NctxComposition
    cpu_only: bool
    device_id: int | None
    off_argv: tuple[str, ...]
    on_argv: tuple[str, ...]
    comparison: MtpComparison
