from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import BenchmarkObservation, ModelFingerprint, ShardFingerprint, ToolchainFingerprint


def write_observation(path: Path, observation: BenchmarkObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _observation_to_dict(observation)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _observation_to_dict(obs: BenchmarkObservation) -> dict[str, object]:
    return {
        "version": 2,
        "candidate": asdict(obs.candidate),
        "total_ctx": obs.total_ctx,
        "prompt_ctx": obs.prompt_ctx,
        "generation_ctx": obs.generation_ctx,
        "depth_ctx": obs.depth_ctx,
        "samples_tps": list(obs.samples_tps),
        "median_tps": obs.median_tps,
        "dispersion_tps": obs.dispersion_tps,
        "n_measurements": len(obs.samples_tps),
        "binary_path": str(obs.binary_path),
        "binary_source": obs.binary_source,
        "model_path": str(obs.model_path),
        "toolchain": _toolchain_to_dict(obs.toolchain),
        "model": _model_to_dict(obs.model),
    }


def _toolchain_to_dict(tc: ToolchainFingerprint) -> dict[str, object]:
    return {
        "binary_path": str(tc.binary_path),
        "version": tc.version,
        "supported_flags": list(tc.supported_flags),
        "supported_output_formats": list(tc.supported_output_formats),
        "device_capabilities": list(tc.device_capabilities),
    }


def _shard_to_dict(s: ShardFingerprint) -> dict[str, object]:
    return {
        "path": str(s.path),
        "size_bytes": s.size_bytes,
        "modified_time": s.modified_time,
        "digest_sha256": s.digest_sha256,
    }


def _model_to_dict(m: ModelFingerprint) -> dict[str, object]:
    return {
        "path": str(m.path),
        "architecture": m.architecture,
        "parameter_count": m.parameter_count,
        "context_length": m.context_length,
        "quantization": m.quantization,
        "shards": [_shard_to_dict(s) for s in m.shards],
        "symlink_warning": m.symlink_warning,
        "network_mount_detected": m.network_mount_detected,
        "diagnostics": list(m.diagnostics),
    }
