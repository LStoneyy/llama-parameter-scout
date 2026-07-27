from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from typing import Any

from .journal import write_observation
from .model_inspector import inspect_model
from .models import (
    BenchmarkObservation,
    BinaryNotFoundError,
    CandidateError,
    InvalidNctxError,
    JournalError,
    MalformedOutputError,
    ModelFingerprint,
    ModelNotFoundError,
    SingleRunRequest,
    SubprocessError,
    ToolchainFingerprint,
)
from .runner import ProcessRunner
from .toolchain import inspect_toolchain


def _build_argv(request: SingleRunRequest, *, repetitions: int) -> list[str]:
    return [
        str(request.binary_path),
        "-m", str(request.model_path),
        "-p", str(request.n_ctx.prompt),
        "-n", str(request.n_ctx.generation),
        "-d", str(request.n_ctx.depth),
        "-t", str(request.candidate.threads),
        "-b", str(request.candidate.batch_size),
        "-ub", str(request.candidate.ubatch_size),
        "-ngl", str(request.candidate.n_gpu_layers),
        "-fa", "on" if request.candidate.flash_attn else "off",
        "-ctk", request.candidate.cache_type_k,
        "-ctv", request.candidate.cache_type_v,
        "-o", "json",
        "-r", str(repetitions),
    ]


def _validate_request(request: SingleRunRequest) -> None:
    bp = request.binary_path
    if not bp.exists():
        raise BinaryNotFoundError(f"Binary not found: {bp}")
    if not bp.is_file():
        raise BinaryNotFoundError(f"Binary is not a file: {bp}")
    if not os.access(str(bp), os.X_OK):
        raise BinaryNotFoundError(f"Binary is not executable: {bp}")

    mp = request.model_path
    if not mp.exists():
        raise ModelNotFoundError(f"Model not found: {mp}")
    if not mp.is_file():
        raise ModelNotFoundError(f"Model is not a file: {mp}")

    nc = request.n_ctx
    if nc.total <= 0:
        raise InvalidNctxError("n_ctx.total must be > 0")
    if nc.prompt < 0:
        raise InvalidNctxError("n_ctx.prompt must be >= 0")
    if nc.generation < 0:
        raise InvalidNctxError("n_ctx.generation must be >= 0")
    if nc.depth < 0:
        raise InvalidNctxError("n_ctx.depth must be >= 0")
    if nc.prompt + nc.generation + nc.depth != nc.total:
        raise InvalidNctxError(
            f"n_ctx components ({nc.prompt} + {nc.generation} + {nc.depth}) "
            f"do not sum to total ({nc.total})",
        )

    cc = request.candidate
    if cc.threads <= 0:
        raise CandidateError("threads must be > 0")
    if cc.batch_size <= 0:
        raise CandidateError("batch_size must be > 0")
    if cc.ubatch_size <= 0:
        raise CandidateError("ubatch_size must be > 0")
    if cc.n_gpu_layers < -1:
        raise CandidateError("n_gpu_layers must be >= -1")


def _parse_measurements(stdout: str) -> list[float]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                pass
        if not data:
            raise MalformedOutputError(
                f"Cannot parse llama-bench output as JSON or JSONL.\n"
                f"Raw output:\n{stdout}",
            )

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise MalformedOutputError(
            f"Top-level JSON is not an array or object.\n"
            f"Raw output:\n{stdout}",
        )

    samples: list[float] | None = None
    for record in data:
        if not isinstance(record, dict):
            continue
        if "samples_ts" not in record:
            continue
        s = record["samples_ts"]
        if not isinstance(s, list) or len(s) == 0:
            continue
        converted: list[float] = []
        for x in s:
            if isinstance(x, bool):
                raise MalformedOutputError(
                    f"Samples contain boolean values (bool must not become number).\n"
                    f"Raw output:\n{stdout}",
                )
            if not isinstance(x, (int, float)):
                raise MalformedOutputError(
                    f"Samples contain non-numeric value: {type(x).__name__}.\n"
                    f"Raw output:\n{stdout}",
                )
            val = float(x)
            if not math.isfinite(val) or val <= 0:
                raise MalformedOutputError(
                    f"samples_ts contains non-finite or non-positive values: {s}\n"
                    f"Raw output:\n{stdout}",
                )
            converted.append(val)
        if len(converted) >= 3:
            samples = converted
            break

    if samples is None or len(samples) < 3:
        raise MalformedOutputError(
            f"Could not find samples_ts with >= 3 entries in llama-bench output.\n"
            f"Raw output:\n{stdout}",
        )

    return samples


def _validate_warmup_output(stdout: str) -> None:
    try:
        json.loads(stdout)
        return
    except json.JSONDecodeError:
        pass
    for line in stdout.strip().splitlines():
        try:
            json.loads(line)
            return
        except json.JSONDecodeError:
            continue
    raise MalformedOutputError(
        f"Warmup output is not valid JSON or JSONL:\n{stdout}",
    )


def _compute_stats(samples: list[float]) -> tuple[float, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n % 2 == 1:
        median = sorted_samples[n // 2]
    else:
        median = (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0

    mean = sum(sorted_samples) / n
    variance = sum((x - mean) ** 2 for x in sorted_samples) / n
    dispersion = math.sqrt(variance)

    return median, dispersion


class Application:
    def __init__(self, runner: ProcessRunner) -> None:
        self._runner = runner

    def run(
        self,
        request: SingleRunRequest,
        *,
        on_ready: Callable[[ToolchainFingerprint, ModelFingerprint], Any] | None = None,
    ) -> BenchmarkObservation:
        _validate_request(request)

        toolchain = inspect_toolchain(self._runner, request.binary_path)
        model = inspect_model(request.model_path)

        if on_ready is not None:
            on_ready(toolchain, model)

        warmup_argv = _build_argv(request, repetitions=1)
        warmup_result = self._runner.run(warmup_argv)
        if warmup_result.returncode != 0:
            raise SubprocessError(
                f"Warmup failed (exit {warmup_result.returncode}):\n"
                f"{warmup_result.stderr}",
            )
        _validate_warmup_output(warmup_result.stdout)

        measure_argv = _build_argv(request, repetitions=3)
        measure_result = self._runner.run(measure_argv)
        if measure_result.returncode != 0:
            raise SubprocessError(
                f"Measurement failed (exit {measure_result.returncode}):\n"
                f"{measure_result.stderr}",
            )

        samples = _parse_measurements(measure_result.stdout)
        median, dispersion = _compute_stats(samples)

        observation = BenchmarkObservation(
            candidate=request.candidate,
            total_ctx=request.n_ctx.total,
            prompt_ctx=request.n_ctx.prompt,
            generation_ctx=request.n_ctx.generation,
            depth_ctx=request.n_ctx.depth,
            samples_tps=tuple(samples),
            median_tps=median,
            dispersion_tps=dispersion,
            binary_path=request.binary_path,
            binary_source=request.binary_source,
            model_path=request.model_path,
            toolchain=toolchain,
            model=model,
        )

        try:
            write_observation(request.journal_path, observation)
        except OSError as e:
            raise JournalError(f"Failed to write journal: {e}") from e

        return observation
