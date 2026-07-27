from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import BenchmarkObservation


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
        "version": 1,
        "candidate": asdict(obs.candidate),
        "total_ctx": obs.total_ctx,
        "prompt_ctx": obs.prompt_ctx,
        "generation_ctx": obs.generation_ctx,
        "depth_ctx": obs.depth_ctx,
        "samples_tps": list(obs.samples_tps),
        "median_tps": obs.median_tps,
        "dispersion_tps": obs.dispersion_tps,
        "n_measurements": len(obs.samples_tps),
    }
