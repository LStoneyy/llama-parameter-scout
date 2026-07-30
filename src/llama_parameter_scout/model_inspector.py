from __future__ import annotations

import glob
import hashlib
import os
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .models import ModelFingerprint, ModelRejectedError, ShardFingerprint

# ── GGUF constants ───────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"

GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

_MAX_STRING_LENGTH = 10 * 1024 * 1024
_MAX_ARRAY_LENGTH = 1_000_000
_MAX_KV_COUNT = 10 * 1024


class _GGUFReadBudget:
    __slots__ = (
        "_wire_bytes_remaining",
        "_value_count_remaining",
        "_max_array_nesting",
        "_max_array_length",
        "_array_depth",
    )

    def __init__(
        self,
        wire_bytes: int = 128 * 1024 * 1024,
        max_values: int = 2_000_000,
        max_array_nesting: int = 16,
        max_array_length: int = _MAX_ARRAY_LENGTH,
    ) -> None:
        self._wire_bytes_remaining = wire_bytes
        self._value_count_remaining = max_values
        self._max_array_nesting = max_array_nesting
        self._max_array_length = max_array_length
        self._array_depth = 0

    def reserve_bytes(self, n: int) -> None:
        if self._wire_bytes_remaining < n:
            raise ModelRejectedError("Metadata byte budget exceeded")
        self._wire_bytes_remaining -= n

    def count_value(self) -> None:
        if self._value_count_remaining <= 0:
            raise ModelRejectedError("Metadata element budget exceeded")
        self._value_count_remaining -= 1

    def enter_array(self, element_count: int) -> None:
        if self._array_depth >= self._max_array_nesting:
            raise ModelRejectedError("Array nesting depth exceeded")
        if element_count >= self._value_count_remaining:
            raise ModelRejectedError("Metadata element budget exceeded")
        self._array_depth += 1

    def leave_array(self) -> None:
        self._array_depth -= 1


_FILE_TYPE_NAMES: dict[int, str] = {
    0: "ALL_F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    6: "Q4_2",
    7: "Q4_3",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q8_0",
    11: "Q8_1",
    12: "Q2_K",
    13: "Q3_K_S",
    14: "Q3_K_M",
    15: "Q3_K_L",
    16: "Q4_K_S",
    17: "Q4_K_M",
    18: "Q5_K_S",
    19: "Q5_K_M",
    20: "Q6_K",
    21: "Q8_K",
    22: "IQ1_S",
    23: "IQ1_M",
    24: "IQ2_XXS",
    25: "IQ2_XS",
    26: "IQ2_S",
    27: "IQ2_M",
    28: "IQ3_XXS",
    29: "IQ3_XS",
    30: "IQ3_S",
    31: "IQ3_M",
    32: "IQ4_NL",
    33: "IQ4_XS",
}

_SHARD_PATTERN = re.compile(r"^(.+)-(\d+)-of-(\d+)\.gguf$")

_URL_SCHEMES = frozenset({"http", "https", "ftp", "s3", "gs", "r2"})

_NetworkMountChecker = Callable[[Path], bool]


def _default_network_mount_checker(path: Path) -> bool:
    resolved = path.resolve()
    if os.name == "nt":
        try:
            anchor = resolved.anchor
            if anchor.startswith("\\\\"):
                return True
        except (ValueError, RuntimeError):
            pass
        return False
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = parts[1]
                fstype = parts[2]
                if fstype in ("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"):
                    try:
                        mp = Path(mount_point)
                        if resolved == mp or mp in resolved.parents:
                            return True
                    except (OSError, RuntimeError):
                        continue
    except (FileNotFoundError, OSError):
        pass
    return False


_network_mount_checker: _NetworkMountChecker = _default_network_mount_checker


def set_network_mount_checker(checker: _NetworkMountChecker | None) -> None:
    global _network_mount_checker
    _network_mount_checker = checker or _default_network_mount_checker


# ── Public API ───────────────────────────────────────────────────


def inspect_model(path: Path | str) -> ModelFingerprint:
    diagnostics: list[str] = []

    if isinstance(path, str):
        if _is_url(path):
            raise ModelRejectedError(
                f"Remote URLs are not supported: {path}. "
                "Provide a local file path.",
            )
        path = Path(path)

    path_str = str(path)
    if _is_url(path_str):
        raise ModelRejectedError(
            f"Remote URLs are not supported: {path}. "
            "Provide a local file path.",
        )

    ext = path.suffix.lower()
    if ext == ".mmproj":
        raise ModelRejectedError(
            f"mmproj files are not supported: {path}. "
            "Only GGUF model files are accepted.",
        )
    if ext == ".lora":
        raise ModelRejectedError(
            f"LoRA adapter files are not supported: {path}. "
            "Only GGUF model files are accepted.",
        )
    if ext != ".gguf":
        raise ModelRejectedError(
            f"Unsupported format: {path}. Only .gguf files are accepted.",
        )

    if not path.exists():
        raise ModelRejectedError(f"Model file not found: {path}")

    metadata = _read_gguf_metadata(path)

    # GGUF adapter detection
    general_type = metadata.get("general.type")
    if isinstance(general_type, str) and general_type.lower() == "adapter":
        adapter_type = metadata.get("adapter.type")
        if isinstance(adapter_type, str) and adapter_type.lower() == "lora":
            raise ModelRejectedError(
                f"GGUF LoRA adapter files are not supported: {path}. "
                "Only GGUF model files are accepted.",
            )
        raise ModelRejectedError(
            f"GGUF adapter files are not supported: {path}."
            + (f" (adapter type: {adapter_type})" if isinstance(adapter_type, str) else "")
            + " Only GGUF model files are accepted.",
        )

    arch = metadata.get("general.architecture")
    if arch is None or not isinstance(arch, str):
        raise ModelRejectedError(
            f"GGUF file missing required 'general.architecture' metadata: {path}",
        )

    param_count: int | None = None
    if "general.parameter_count" in metadata:
        pc = metadata["general.parameter_count"]
        if isinstance(pc, (int, float)):
            param_count = int(pc)
        else:
            diagnostics.append(
                f"general.parameter_count has unexpected type {type(pc).__name__}, "
                "treating as unknown",
            )

    ctx_length: int | None = None
    ctx_key = f"{arch}.context_length"
    ctx_raw = metadata.get(ctx_key)
    if ctx_raw is not None:
        if isinstance(ctx_raw, (int, float)):
            ctx_length = int(ctx_raw)
        else:
            diagnostics.append(
                f"{ctx_key} has unexpected type {type(ctx_raw).__name__}, "
                "treating as unknown",
            )
    else:
        diagnostics.append(f"{ctx_key} not found, context length unknown")

    quantization: str | None = None
    ft = metadata.get("general.file_type")
    if ft is not None:
        if isinstance(ft, int):
            quantization = _FILE_TYPE_NAMES.get(ft, f"qtype_{ft}")
        else:
            diagnostics.append(
                "general.file_type has unexpected type, treating as unknown",
            )
    else:
        diagnostics.append("general.file_type not found, quantization unknown")

    shards = _resolve_shards(path, diagnostics)

    symlink_warning = False
    network_mount_detected = False
    if shards:
        all_shard_paths = [s.path for s in shards]
        for shard_path in all_shard_paths:
            if _has_symlink_ancestor(shard_path):
                if not symlink_warning:
                    diagnostics.append(f"Path involves symlink(s): {shard_path}")
                symlink_warning = True
            if _network_mount_checker(shard_path):
                if not network_mount_detected:
                    diagnostics.append(f"Path appears to be on a network mount: {shard_path}")
                network_mount_detected = True

    is_moe = False
    moe_expert_count: int | None = None
    expert_count_key = f"{arch}.expert_count"
    if expert_count_key in metadata:
        ec = metadata[expert_count_key]
        if isinstance(ec, (int, float)):
            is_moe = True
            moe_expert_count = int(ec)
        else:
            diagnostics.append(
                f"{expert_count_key} has unexpected type {type(ec).__name__}, "
                "treating as non-MoE",
            )

    if len(shards) > 1:
        shard_ctx_values: set[int | None] = set()
        for s in shards:
            if s.path == path:
                meta = metadata
            else:
                meta = _read_gguf_metadata(s.path)
            raw = meta.get(ctx_key)
            if isinstance(raw, (int, float)):
                shard_ctx_values.add(int(raw))
            else:
                shard_ctx_values.add(None)
        if len(shard_ctx_values) > 1:
            diagnostics.append(
                f"Shard context length conflict: values {shard_ctx_values} "
                f"across shards for {ctx_key}. Treating as unknown."
            )
            ctx_length = None

    return ModelFingerprint(
        path=path,
        architecture=arch,
        parameter_count=param_count,
        context_length=ctx_length,
        quantization=quantization,
        shards=tuple(shards),
        symlink_warning=symlink_warning,
        network_mount_detected=network_mount_detected,
        diagnostics=tuple(diagnostics),
        is_moe=is_moe,
        moe_expert_count=moe_expert_count,
    )


# ── URL check ────────────────────────────────────────────────────


def _is_url(path_str: str) -> bool:
    if "://" not in path_str:
        return False
    scheme = path_str.split("://", 1)[0].lower()
    return scheme in _URL_SCHEMES


# ── Symlink detection ────────────────────────────────────────────


def _has_symlink_ancestor(path: Path) -> bool:
    if path.is_symlink():
        return True
    for parent in path.parents:
        if parent.is_symlink():
            return True
    return False


# ── Shard resolution ─────────────────────────────────────────────


def _resolve_shards(primary: Path, diagnostics: list[str]) -> list[ShardFingerprint]:
    shard_match = _SHARD_PATTERN.match(primary.name)
    if not shard_match:
        fp = _fingerprint_file(primary)
        if fp is None:
            raise ModelRejectedError(
                f"Cannot fingerprint model file: {primary}",
            )
        return [fp]

    prefix = shard_match.group(1)
    shard_num_str = shard_match.group(2)
    total_str = shard_match.group(3)
    total = int(total_str)
    width = len(shard_num_str)
    total_width = len(total_str)

    directory = primary.parent
    shards_list: list[ShardFingerprint] = []
    seen_paths: set[Path] = set()
    primary_metadata = _read_gguf_metadata(primary)
    primary_arch = primary_metadata.get("general.architecture")

    for i in range(1, total + 1):
        shard_name = f"{prefix}-{i:0{width}d}-of-{total:0{total_width}d}.gguf"
        shard_path = directory / shard_name
        if not shard_path.exists():
            raise ModelRejectedError(
                f"Missing shard {shard_name} (expected from {primary.name})",
            )
        fp = _fingerprint_file(shard_path)
        if fp is None:
            raise ModelRejectedError(
                f"Cannot fingerprint shard file: {shard_path}",
            )
        shards_list.append(fp)
        seen_paths.add(shard_path)

        if shard_path == primary:
            shard_meta = primary_metadata
        else:
            shard_meta = _read_gguf_metadata(shard_path)
            shard_arch = shard_meta.get("general.architecture")
            if shard_arch is not None and primary_arch is not None:
                if shard_arch != primary_arch:
                    raise ModelRejectedError(
                        f"Architecture mismatch in shard {shard_name}: "
                        f"expected {primary_arch}, got {shard_arch}",
                    )

        shard_split_no = shard_meta.get("split.no")
        if shard_split_no is not None and isinstance(shard_split_no, int):
            if shard_split_no != i - 1:
                raise ModelRejectedError(
                    f"split.no={shard_split_no} in {shard_name} "
                    f"does not match shard index {i}: {shard_path}",
                )

        shard_split_count = shard_meta.get("split.count")
        if shard_split_count is not None and isinstance(shard_split_count, int):
            if shard_split_count != total:
                raise ModelRejectedError(
                    f"split.count={shard_split_count} in {shard_name} "
                    f"does not match filename total={total}: {shard_path}",
                )

    glob_pattern = f"{glob.escape(prefix)}-*-of-{total:0{total_width}d}.gguf"
    for extra in directory.glob(glob_pattern):
        if extra not in seen_paths:
            raise ModelRejectedError(
                f"Unexpected extra shard file found: {extra}",
            )

    return shards_list


def _fingerprint_file(path: Path) -> ShardFingerprint | None:
    try:
        st = path.stat()
        digest = _sha256(path)
        return ShardFingerprint(
            path=path,
            size_bytes=st.st_size,
            modified_time=st.st_mtime,
            digest_sha256=digest,
        )
    except OSError:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── GGUF metadata reader ─────────────────────────────────────────


def _read_gguf_metadata(
    path: Path,
    _budget: _GGUFReadBudget | None = None,
) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            budget = _GGUFReadBudget() if _budget is None else _budget
            return _parse_gguf(f, path, budget)
    except (OSError, struct.error, UnicodeDecodeError, EOFError) as e:
        raise ModelRejectedError(
            f"Failed to read GGUF metadata from {path}: {e}",
        ) from e


def _parse_gguf(f: BinaryIO, path: Path, budget: _GGUFReadBudget) -> dict[str, Any]:
    magic = _read_exact(f, 4, budget)
    if magic != GGUF_MAGIC:
        raise ModelRejectedError(
            f"Not a GGUF file (bad magic: {magic!r}): {path}",
        )

    version = struct.unpack("<I", _read_exact(f, 4, budget))[0]
    if version not in (2, 3):
        raise ModelRejectedError(
            f"Unsupported GGUF version {version}. "
            "Only versions 2 and 3 are supported.",
        )

    _tensor_count = struct.unpack("<Q", _read_exact(f, 8, budget))[0]
    kv_count = struct.unpack("<Q", _read_exact(f, 8, budget))[0]

    if kv_count > _MAX_KV_COUNT:
        raise ModelRejectedError(
            f"GGUF metadata KV count {kv_count} exceeds maximum {_MAX_KV_COUNT}: {path}",
        )

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _read_gguf_string(f, path, budget)
        value_type = struct.unpack("<I", _read_exact(f, 4, budget))[0]
        value = _read_gguf_value(f, value_type, path, budget)
        metadata[key] = value

    return metadata


def _read_exact(f: BinaryIO, n: int, budget: _GGUFReadBudget) -> bytes:
    budget.reserve_bytes(n)
    data = f.read(n)
    if len(data) < n:
        raise EOFError(
            f"Expected {n} bytes, got {len(data)} — file truncated",
        )
    return data


def _read_gguf_string(f: BinaryIO, path: Path, budget: _GGUFReadBudget) -> str:
    length = struct.unpack("<Q", _read_exact(f, 8, budget))[0]
    if length > _MAX_STRING_LENGTH:
        raise ModelRejectedError(
            f"GGUF string length {length} exceeds maximum {_MAX_STRING_LENGTH}: {path}",
        )
    raw = _read_exact(f, length, budget)
    budget.count_value()
    return raw.decode("utf-8")


def _read_gguf_value(
    f: BinaryIO,
    value_type: int,
    path: Path,
    budget: _GGUFReadBudget,
) -> Any:
    if value_type == GGUF_VALUE_TYPE_UINT8:
        budget.count_value()
        return struct.unpack("<B", _read_exact(f, 1, budget))[0]
    if value_type == GGUF_VALUE_TYPE_INT8:
        budget.count_value()
        return struct.unpack("<b", _read_exact(f, 1, budget))[0]
    if value_type == GGUF_VALUE_TYPE_UINT16:
        budget.count_value()
        return struct.unpack("<H", _read_exact(f, 2, budget))[0]
    if value_type == GGUF_VALUE_TYPE_INT16:
        budget.count_value()
        return struct.unpack("<h", _read_exact(f, 2, budget))[0]
    if value_type == GGUF_VALUE_TYPE_UINT32:
        budget.count_value()
        return struct.unpack("<I", _read_exact(f, 4, budget))[0]
    if value_type == GGUF_VALUE_TYPE_INT32:
        budget.count_value()
        return struct.unpack("<i", _read_exact(f, 4, budget))[0]
    if value_type == GGUF_VALUE_TYPE_FLOAT32:
        budget.count_value()
        return struct.unpack("<f", _read_exact(f, 4, budget))[0]
    if value_type == GGUF_VALUE_TYPE_BOOL:
        budget.count_value()
        return bool(struct.unpack("<B", _read_exact(f, 1, budget))[0])
    if value_type == GGUF_VALUE_TYPE_STRING:
        return _read_gguf_string(f, path, budget)
    if value_type == GGUF_VALUE_TYPE_ARRAY:
        elem_type = struct.unpack("<I", _read_exact(f, 4, budget))[0]
        length = struct.unpack("<Q", _read_exact(f, 8, budget))[0]
        if length > budget._max_array_length:
            raise ModelRejectedError(
                f"GGUF array length {length} exceeds maximum {budget._max_array_length}: {path}",
            )
        budget.enter_array(length)
        budget.count_value()
        try:
            result = [
                _read_gguf_value(f, elem_type, path, budget) for _ in range(length)
            ]
        finally:
            budget.leave_array()
        return result
    if value_type == GGUF_VALUE_TYPE_UINT64:
        budget.count_value()
        return struct.unpack("<Q", _read_exact(f, 8, budget))[0]
    if value_type == GGUF_VALUE_TYPE_INT64:
        budget.count_value()
        return struct.unpack("<q", _read_exact(f, 8, budget))[0]
    if value_type == GGUF_VALUE_TYPE_FLOAT64:
        budget.count_value()
        return struct.unpack("<d", _read_exact(f, 8, budget))[0]
    raise ModelRejectedError(f"Unknown GGUF metadata value type: {value_type}")
