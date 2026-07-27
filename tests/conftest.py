from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    ModelFingerprint,
    NctxComposition,
    SingleRunRequest,
    ToolchainFingerprint,
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
    _build_minimal_gguf(p, {"general.architecture": (8, "llama")})
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
        binary_source="explicit",
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
def valid_version_output() -> str:
    return (
        "version: 1234\n"
        "build_commit: abcdef1234567890abcdef1234567890abcdef12\n"
        "build_info: built with Apple Clang 14.0.0 for arm64-apple-macosx14.0\n"
        "backends: cpu,metal\n"
        "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
        "supported_output_formats: json,jsonl\n"
    )


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def sample_toolchain() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/usr/local/bin/llama-bench"),
        version="1234",
        supported_flags=(
            "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
            "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
        ),
        supported_output_formats=("json", "jsonl"),
        device_capabilities=("cpu", "metal"),
    )


@pytest.fixture
def sample_model() -> ModelFingerprint:
    return ModelFingerprint(
        path=Path("/tmp/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


# ── GGUF builder for tests ────────────────────────────────────────


def _build_minimal_gguf(path: Path, metadata: dict[str, tuple[int, Any]], version: int = 3) -> None:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", version))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", len(metadata)))
    for key, (value_type, value) in metadata.items():
        _pack_gguf_string(buf, key)
        buf.extend(struct.pack("<I", value_type))
        _pack_gguf_value(buf, value_type, value)
    path.write_bytes(bytes(buf))


def _pack_gguf_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _pack_gguf_value(buf: bytearray, value_type: int, value: Any) -> None:
    if value_type == 0:
        buf.extend(struct.pack("<B", value))
    elif value_type == 1:
        buf.extend(struct.pack("<b", value))
    elif value_type == 2:
        buf.extend(struct.pack("<H", value))
    elif value_type == 3:
        buf.extend(struct.pack("<h", value))
    elif value_type == 4:
        buf.extend(struct.pack("<I", value))
    elif value_type == 5:
        buf.extend(struct.pack("<i", value))
    elif value_type == 6:
        buf.extend(struct.pack("<f", value))
    elif value_type == 7:
        buf.extend(struct.pack("<B", 1 if value else 0))
    elif value_type == 8:
        _pack_gguf_string(buf, value)
    elif value_type == 9:
        elem_type, elements = value
        buf.extend(struct.pack("<I", elem_type))
        buf.extend(struct.pack("<Q", len(elements)))
        for elem in elements:
            _pack_gguf_value(buf, elem_type, elem)
    elif value_type == 10:
        buf.extend(struct.pack("<Q", value))
    elif value_type == 11:
        buf.extend(struct.pack("<q", value))
    elif value_type == 12:
        buf.extend(struct.pack("<d", value))
