from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest

from llama_parameter_scout.model_inspector import (
    _GGUFReadBudget,
    _has_symlink_ancestor,
    _is_url,
    _parse_gguf,
    _read_gguf_metadata,
    inspect_model,
    set_network_mount_checker,
)
from llama_parameter_scout.models import ModelRejectedError
from tests.conftest import _build_minimal_gguf as build_gguf


class TestInspectModel:
    def test_single_gguf(self, tmp_path: Path) -> None:
        gguf_path = tmp_path / "model.gguf"
        build_gguf(gguf_path, {
            "general.architecture": (8, "llama"),
            "general.file_type": (5, 17),
            "general.parameter_count": (10, 7_000_000_000),
            "llama.context_length": (10, 4096),
        })
        fp = inspect_model(gguf_path)
        assert fp.architecture == "llama"
        assert fp.parameter_count == 7_000_000_000
        assert fp.context_length == 4096
        assert fp.quantization == "Q4_K_M"
        assert len(fp.shards) == 1
        assert not fp.symlink_warning
        assert not fp.network_mount_detected
        assert fp.diagnostics == ()

    def test_single_gguf_shard_fingerprint(self, tmp_path: Path) -> None:
        gguf_path = tmp_path / "model.gguf"
        build_gguf(gguf_path, {"general.architecture": (8, "llama")})
        fp = inspect_model(gguf_path)
        assert len(fp.shards) == 1
        s = fp.shards[0]
        assert s.path == gguf_path
        assert s.size_bytes > 0
        assert s.modified_time > 0
        assert len(s.digest_sha256) == 64

    def test_unknown_optional_metadata(self, tmp_path: Path) -> None:
        gguf_path = tmp_path / "model.gguf"
        build_gguf(gguf_path, {"general.architecture": (8, "llama")})
        fp = inspect_model(gguf_path)
        assert fp.architecture == "llama"
        assert fp.parameter_count is None
        assert fp.context_length is None
        assert fp.quantization is None
        assert len(fp.diagnostics) >= 2

    def test_reject_url(self) -> None:
        with pytest.raises(ModelRejectedError, match="Remote URLs"):
            inspect_model("https://example.com/model.gguf")
        with pytest.raises(ModelRejectedError, match="Remote URLs"):
            inspect_model("s3://bucket/model.gguf")

    def test_reject_mmproj(self, tmp_path: Path) -> None:
        p = tmp_path / "clip.mmproj"
        p.write_text("data")
        with pytest.raises(ModelRejectedError, match="mmproj"):
            inspect_model(p)

    def test_reject_lora_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "adapter.lora"
        p.write_text("data")
        with pytest.raises(ModelRejectedError, match="LoRA"):
            inspect_model(p)

    def test_reject_gguf_lora_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "lora_adapter.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "general.type": (8, "adapter"),
            "adapter.type": (8, "lora"),
        })
        with pytest.raises(ModelRejectedError, match="LoRA adapter"):
            inspect_model(p)

    def test_reject_gguf_adapter_without_type(self, tmp_path: Path) -> None:
        p = tmp_path / "adapter.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "general.type": (8, "adapter"),
        })
        with pytest.raises(ModelRejectedError, match="adapter"):
            inspect_model(p)

    def test_reject_gguf_lora_case_insensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "lora.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "general.type": (8, "Adapter"),
            "adapter.type": (8, "LoRA"),
        })
        with pytest.raises(ModelRejectedError, match="LoRA adapter"):
            inspect_model(p)

    def test_reject_unsupported_format(self, tmp_path: Path) -> None:
        p = tmp_path / "model.bin"
        p.write_text("data")
        with pytest.raises(ModelRejectedError, match="Unsupported format"):
            inspect_model(p)

    def test_reject_non_gguf_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        p.write_text("NOT A GGUF FILE")
        with pytest.raises(ModelRejectedError, match="Not a GGUF"):
            inspect_model(p)

    def test_reject_missing_architecture(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"general.file_type": (5, 2)})
        with pytest.raises(ModelRejectedError, match="missing required"):
            inspect_model(p)

    def test_shard_set(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00003.gguf"
        for i in range(1, 4):
            shard = tmp_path / f"model-{i:05d}-of-00003.gguf"
            build_gguf(shard, {"general.architecture": (8, "llama")})
            with open(shard, "ab") as f:
                f.write(struct.pack("<i", i))
        fp = inspect_model(base)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3
        for s in fp.shards:
            assert s.size_bytes > 0
            assert s.modified_time > 0
            assert len(s.digest_sha256) == 64

    def test_missing_shard_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00003.gguf"
        build_gguf(base, {"general.architecture": (8, "llama")})
        s3 = tmp_path / "model-00003-of-00003.gguf"
        build_gguf(s3, {"general.architecture": (8, "llama")})
        with pytest.raises(ModelRejectedError, match="Missing shard"):
            inspect_model(base)

    def test_architecture_mismatch_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {"general.architecture": (8, "llama")})
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {"general.architecture": (8, "bert")})
        with pytest.raises(ModelRejectedError, match="Architecture mismatch"):
            inspect_model(base)

    def test_split_count_mismatch_rejected(self, tmp_path: Path) -> None:
        for i in range(1, 4):
            shard = tmp_path / f"model-{i:05d}-of-00003.gguf"
            build_gguf(shard, {"general.architecture": (8, "llama")})
        base = tmp_path / "model-00001-of-00003.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "split.count": (5, 2),
        })
        with pytest.raises(ModelRejectedError, match="split.count"):
            inspect_model(base)

    def test_split_no_mismatch_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {"general.architecture": (8, "llama")})
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 99),
        })
        with pytest.raises(ModelRejectedError, match="split.no"):
            inspect_model(base)

    def test_split_no_first_shard_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 42),
        })
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {"general.architecture": (8, "llama")})
        with pytest.raises(ModelRejectedError, match="split.no.*42"):
            inspect_model(base)

    def test_valid_split_no_zero_based(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 0),
            "split.count": (5, 2),
        })
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 1),
            "split.count": (5, 2),
        })
        fp = inspect_model(base)
        assert len(fp.shards) == 2
        assert fp.architecture == "llama"

    def test_shard_context_length_conflict_detected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "llama.context_length": (10, 4096),
            "split.no": (5, 0),
            "split.count": (5, 2),
        })
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {
            "general.architecture": (8, "llama"),
            "llama.context_length": (10, 8192),
            "split.no": (5, 1),
            "split.count": (5, 2),
        })
        fp = inspect_model(base)
        assert fp.context_length is None
        assert any("conflict" in d for d in fp.diagnostics)

    def test_shard_context_length_agreement(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "llama.context_length": (10, 4096),
            "split.no": (5, 0),
            "split.count": (5, 2),
        })
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {
            "general.architecture": (8, "llama"),
            "llama.context_length": (10, 4096),
            "split.no": (5, 1),
            "split.count": (5, 2),
        })
        fp = inspect_model(base)
        assert fp.context_length == 4096

    def test_split_count_non_primary_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "model-00001-of-00002.gguf"
        build_gguf(base, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 0),
            "split.count": (5, 2),
        })
        s2 = tmp_path / "model-00002-of-00002.gguf"
        build_gguf(s2, {
            "general.architecture": (8, "llama"),
            "split.no": (5, 1),
            "split.count": (5, 99),
        })
        with pytest.raises(ModelRejectedError, match="split.count"):
            inspect_model(base)

    def test_extra_shard_rejected(self, tmp_path: Path) -> None:
        for i in range(1, 4):
            shard = tmp_path / f"model-{i:05d}-of-00002.gguf"
            build_gguf(shard, {"general.architecture": (8, "llama")})
        base = tmp_path / "model-00001-of-00002.gguf"
        with pytest.raises(ModelRejectedError, match="extra shard"):
            inspect_model(base)

    def test_symlink_warning(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_gguf = real_dir / "model.gguf"
        build_gguf(real_gguf, {"general.architecture": (8, "llama")})
        link = tmp_path / "link.gguf"
        link.symlink_to(real_gguf)
        fp = inspect_model(link)
        assert fp.symlink_warning
        assert any("symlink" in d for d in fp.diagnostics)

    def test_symlink_warning_shard(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        for i in range(1, 3):
            shard = real_dir / f"model-{i:05d}-of-00002.gguf"
            build_gguf(shard, {"general.architecture": (8, "llama")})
        primary_link = tmp_path / "model-00001-of-00002.gguf"
        primary_link.symlink_to(real_dir / "model-00001-of-00002.gguf")
        second_link = tmp_path / "model-00002-of-00002.gguf"
        second_link.symlink_to(real_dir / "model-00002-of-00002.gguf")
        fp = inspect_model(primary_link)
        assert fp.symlink_warning
        assert any("symlink" in d for d in fp.diagnostics)

    def test_network_mount_warning(self, tmp_path: Path) -> None:
        gguf = tmp_path / "model.gguf"
        build_gguf(gguf, {"general.architecture": (8, "llama")})
        set_network_mount_checker(lambda p: True)
        try:
            fp = inspect_model(gguf)
            assert fp.network_mount_detected
            assert any("network mount" in d for d in fp.diagnostics)
        finally:
            set_network_mount_checker(None)

    def test_model_file_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.gguf"
        with pytest.raises(ModelRejectedError, match="not found"):
            inspect_model(p)

    def test_unknown_file_type(self, tmp_path: Path) -> None:
        gguf = tmp_path / "model.gguf"
        build_gguf(gguf, {
            "general.architecture": (8, "llama"),
            "general.file_type": (5, 999),
        })
        fp = inspect_model(gguf)
        assert fp.quantization == "qtype_999"
        assert "qtype_999" not in fp.diagnostics

    def test_wrong_file_type_non_integer(self, tmp_path: Path) -> None:
        gguf = tmp_path / "model.gguf"
        build_gguf(gguf, {
            "general.architecture": (8, "llama"),
            "general.file_type": (8, "unknown"),
        })
        fp = inspect_model(gguf)
        assert fp.quantization is None
        assert any("file_type" in d and "unexpected type" in d for d in fp.diagnostics)

    def test_truncated_gguf(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        buf = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 100)
        p.write_bytes(buf)
        with pytest.raises(ModelRejectedError, match="truncated|Failed to read"):
            inspect_model(p)

    def test_absurd_kv_count(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        buf = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 999_999)
        p.write_bytes(buf)
        with pytest.raises(ModelRejectedError, match="exceeds maximum"):
            inspect_model(p)

    def test_absurd_string_length(self, tmp_path: Path) -> None:
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<I", 3))
        buf.extend(struct.pack("<Q", 0))
        buf.extend(struct.pack("<Q", 1))
        _pack_gguf_string_raw(buf, "k")
        buf.extend(struct.pack("<I", 8))
        buf.extend(struct.pack("<Q", 99_999_999))
        p = tmp_path / "model.gguf"
        p.write_bytes(bytes(buf))
        with pytest.raises(ModelRejectedError, match="exceeds maximum"):
            inspect_model(p)

    def test_unknown_value_type(self, tmp_path: Path) -> None:
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<I", 3))
        buf.extend(struct.pack("<Q", 0))
        buf.extend(struct.pack("<Q", 1))
        _pack_gguf_string_raw(buf, "k")
        buf.extend(struct.pack("<I", 255))
        p = tmp_path / "model.gguf"
        p.write_bytes(bytes(buf))
        with pytest.raises(ModelRejectedError, match="Unknown GGUF metadata value type: 255"):
            inspect_model(p)

    def test_full_metadata_fixture(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "general.file_type": (5, 16),
            "general.parameter_count": (10, 7_000_000_000),
            "llama.context_length": (10, 8192),
            "llama.attention.layer_count": (4, 32),
            "llama.block_count": (4, 32),
        })
        fp = inspect_model(p)
        assert fp.architecture == "llama"
        assert fp.parameter_count == 7_000_000_000
        assert fp.context_length == 8192
        assert fp.quantization == "Q4_K_S"


class TestIsUrl:
    def test_http(self) -> None:
        assert _is_url("https://example.com/model.gguf")

    def test_s3(self) -> None:
        assert _is_url("s3://bucket/model.gguf")

    def test_local_path(self) -> None:
        assert not _is_url("/home/user/model.gguf")

    def test_relative_path(self) -> None:
        assert not _is_url("model.gguf")


class TestHasSymlinkAncestor:
    def test_file_is_symlink(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.write_text("data")
        link = tmp_path / "link"
        link.symlink_to(target)
        assert _has_symlink_ancestor(link)

    def test_no_symlink(self, tmp_path: Path) -> None:
        f = tmp_path / "file.gguf"
        f.write_text("data")
        assert not _has_symlink_ancestor(f)


class TestReadGgufMetadata:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"general.architecture": (8, "llama")})
        meta = _read_gguf_metadata(p)
        assert meta["general.architecture"] == "llama"

    def test_v2(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"general.architecture": (8, "llama")}, version=2)
        meta = _read_gguf_metadata(p)
        assert meta["general.architecture"] == "llama"

    def test_v3(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"general.architecture": (8, "llama")}, version=3)
        meta = _read_gguf_metadata(p)
        assert meta["general.architecture"] == "llama"

    def test_unsupported_version(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"general.architecture": (8, "llama")}, version=1)
        with pytest.raises(ModelRejectedError, match="Unsupported GGUF version"):
            _read_gguf_metadata(p)

    def test_multiple_keys(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "general.file_type": (5, 16),
            "llama.context_length": (10, 4096),
        })
        meta = _read_gguf_metadata(p)
        assert meta["general.architecture"] == "llama"
        assert meta["general.file_type"] == 16
        assert meta["llama.context_length"] == 4096

    def test_array_value(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"test.array": (9, (5, [1, 2, 3]))})
        meta = _read_gguf_metadata(p)
        assert meta["test.array"] == [1, 2, 3]

    def test_absurd_array_length(self, tmp_path: Path) -> None:
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<I", 3))
        buf.extend(struct.pack("<Q", 0))
        buf.extend(struct.pack("<Q", 1))
        _pack_gguf_string_raw(buf, "a")
        buf.extend(struct.pack("<I", 9))
        buf.extend(struct.pack("<I", 5))
        buf.extend(struct.pack("<Q", 99_999_999))
        p = tmp_path / "model.gguf"
        p.write_bytes(bytes(buf))
        with pytest.raises(ModelRejectedError, match="exceeds maximum"):
            _read_gguf_metadata(p)


def _pack_gguf_string_raw(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


class TestGgufReadBudget:
    """Budget enforcement for GGUF metadata parsing."""

    # -- wire byte budget -----------------------------------------------

    def test_byte_budget_exact(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {})
        meta = _read_gguf_metadata(p, _budget=_GGUFReadBudget(wire_bytes=24, max_values=10))
        assert meta == {}

    def test_byte_budget_exceeded(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {})
        with pytest.raises(ModelRejectedError, match="Metadata byte budget exceeded"):
            _read_gguf_metadata(p, _budget=_GGUFReadBudget(wire_bytes=23, max_values=10))

    # -- cumulative element (value) budget ------------------------------

    def test_element_budget_scalar(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"a": (0, 1), "b": (0, 2)})
        meta = _read_gguf_metadata(p, _budget=_GGUFReadBudget(wire_bytes=200, max_values=4))
        assert meta == {"a": 1, "b": 2}

    def test_element_budget_scalar_exceeded(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"a": (0, 1), "b": (0, 2)})
        with pytest.raises(ModelRejectedError, match="Metadata element budget exceeded"):
            _read_gguf_metadata(p, _budget=_GGUFReadBudget(wire_bytes=200, max_values=3))

    def test_element_budget_nested_arrays(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (9, [(5, [42]), (5, [99])]))})
        meta = _read_gguf_metadata(
            p, _budget=_GGUFReadBudget(wire_bytes=500, max_values=6, max_array_nesting=5),
        )
        assert meta == {"arr": [[42], [99]]}

    def test_element_budget_nested_arrays_exceeded(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (9, [(5, [42]), (5, [99])]))})
        with pytest.raises(ModelRejectedError, match="Metadata element budget exceeded"):
            _read_gguf_metadata(
                p, _budget=_GGUFReadBudget(wire_bytes=500, max_values=5, max_array_nesting=5),
            )

    # -- single-array length limit --------------------------------------

    def test_array_length_exact(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (5, [1, 2, 3]))})
        meta = _read_gguf_metadata(
            p, _budget=_GGUFReadBudget(max_array_length=3, wire_bytes=500, max_values=10),
        )
        assert meta == {"arr": [1, 2, 3]}

    def test_array_length_exceeded(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (5, [1, 2, 3, 4]))})
        with pytest.raises(
            ModelRejectedError,
            match=r"GGUF array length 4 exceeds maximum 3",
        ):
            _read_gguf_metadata(
                p, _budget=_GGUFReadBudget(max_array_length=3, wire_bytes=500, max_values=10),
            )

    # -- array nesting depth --------------------------------------------

    def test_nesting_exact(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (9, [(5, [1])]))})
        meta = _read_gguf_metadata(
            p, _budget=_GGUFReadBudget(wire_bytes=500, max_values=10, max_array_nesting=2),
        )
        assert meta == {"arr": [[1]]}

    def test_nesting_exceeded(self, tmp_path: Path) -> None:
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (9, [(9, [(5, [1])])]))})
        with pytest.raises(ModelRejectedError, match="Array nesting depth exceeded"):
            _read_gguf_metadata(
                p, _budget=_GGUFReadBudget(wire_bytes=500, max_values=10, max_array_nesting=2),
            )

    # -- stream-position proof: rejection before element read -----------

    def test_array_element_not_read_when_budget_exceeded(self, tmp_path: Path) -> None:
        """Verify enter_array preflight rejects before reading elements."""
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (5, [10, 20, 30]))})
        raw = p.read_bytes()

        # Position breakdown for array-of-3-int32 metadata:
        #   4 magic + 4 version + 8 tensor_count + 8 kv_count
        # + 8 key_len + 3 "arr" + 4 value_type
        # + 4 elem_type + 8 length
        # = 51 (array element data starts at offset 51)
        array_header_end = 51

        # Success path: budget covers key + container + 3 elements = 5
        stream = io.BytesIO(raw)
        budget = _GGUFReadBudget(
            wire_bytes=500, max_values=5, max_array_nesting=5,
        )
        meta = _parse_gguf(stream, p, budget)
        assert meta == {"arr": [10, 20, 30]}

        # Failure path: budget covers only key + container + 2 elements = 4
        # enter_array (element_count=3, remaining=3) → 3 >= 3 → reject
        stream = io.BytesIO(raw)
        budget = _GGUFReadBudget(
            wire_bytes=500, max_values=4, max_array_nesting=5,
        )
        with pytest.raises(ModelRejectedError, match="Metadata element budget exceeded"):
            _parse_gguf(stream, p, budget)
        assert stream.tell() == array_header_end, (
            f"Stream advanced to {stream.tell()} (expected {array_header_end}); "
            "an array element was read despite budget rejection"
        )

    # -- depth restored after post-enter failure -----------------------

    def test_array_depth_restored_after_element_read_failure(self, tmp_path: Path) -> None:
        """Element read inside try block fails; finally must restore depth."""
        p = tmp_path / "model.gguf"
        build_gguf(p, {"arr": (9, (5, [10, 20, 30]))})
        raw = p.read_bytes()

        budget = _GGUFReadBudget(
            wire_bytes=51, max_values=10, max_array_nesting=5,
        )
        stream = io.BytesIO(raw)
        with pytest.raises(ModelRejectedError, match="Metadata byte budget exceeded"):
            _parse_gguf(stream, p, budget)
        assert budget._array_depth == 0, (
            f"Array depth left at {budget._array_depth} after element read failure"
        )


class TestShardPrefix:
    """Verify glob.escape does not break special-char prefixes.
    Each test places a decoy file whose filename the *unescaped* glob would
    incorrectly match; the test passes only when the escaped glob ignores it.
    """

    def _build_shards(self, tmp_path: Path, prefix: str, total: int) -> Path:
        primary: Path | None = None
        for i in range(1, total + 1):
            name = f"{prefix}-{i:05d}-of-{total:05d}.gguf"
            shard = tmp_path / name
            build_gguf(shard, {"general.architecture": (8, "llama")})
            if i == 1:
                primary = shard
        assert primary is not None
        return primary

    def test_prefix_with_star(self, tmp_path: Path) -> None:
        primary = self._build_shards(tmp_path, "my*-model", 3)
        decoy = tmp_path / "myX-model-00001-of-00003.gguf"
        build_gguf(decoy, {"general.architecture": (8, "llama")})
        fp = inspect_model(primary)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3

    def test_prefix_with_question(self, tmp_path: Path) -> None:
        primary = self._build_shards(tmp_path, "my?-model", 3)
        decoy = tmp_path / "myX-model-00001-of-00003.gguf"
        build_gguf(decoy, {"general.architecture": (8, "llama")})
        fp = inspect_model(primary)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3

    def test_prefix_with_bracket_open(self, tmp_path: Path) -> None:
        primary = self._build_shards(tmp_path, "my[-model", 3)
        decoy = tmp_path / "myX-model-00001-of-00003.gguf"
        build_gguf(decoy, {"general.architecture": (8, "llama")})
        fp = inspect_model(primary)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3

    def test_prefix_with_bracket_close(self, tmp_path: Path) -> None:
        primary = self._build_shards(tmp_path, "my]-model", 3)
        decoy = tmp_path / "myX-model-00001-of-00003.gguf"
        build_gguf(decoy, {"general.architecture": (8, "llama")})
        fp = inspect_model(primary)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3

    def test_normal_shards_unchanged(self, tmp_path: Path) -> None:
        primary = self._build_shards(tmp_path, "model", 3)
        decoy = tmp_path / "myX-model-00001-of-00003.gguf"
        build_gguf(decoy, {"general.architecture": (8, "llama")})
        fp = inspect_model(primary)
        assert fp.architecture == "llama"
        assert len(fp.shards) == 3
