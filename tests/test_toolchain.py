from __future__ import annotations

from pathlib import Path

import pytest

from llama_parameter_scout.models import ToolchainError
from llama_parameter_scout.runner import FakeRunner, ProcessResult
from llama_parameter_scout.toolchain import (
    _parse_backends,
    _parse_output_formats,
    _parse_supported_flags,
    _parse_version,
    inspect_toolchain,
    resolve_binary,
)


class TestResolveBinary:
    def test_explicit_path_used_as_is(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        bp.chmod(0o755)
        result = resolve_binary(llama_bench=bp)
        assert result.binary_path == bp
        assert result.source == "explicit"

    def test_explicit_overrides_install_dir(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        bp.chmod(0o755)
        install = tmp_path / "install"
        install.mkdir()
        (install / "llama-bench").touch()
        (install / "llama-bench").chmod(0o755)
        result = resolve_binary(llama_bench=bp, install_dir=install)
        assert result.binary_path == bp
        assert result.source == "explicit"

    def test_explicit_overrides_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        bp.chmod(0o755)
        path_dir = tmp_path / "pathdir"
        path_dir.mkdir()
        (path_dir / "llama-bench").touch()
        (path_dir / "llama-bench").chmod(0o755)
        monkeypatch.setenv("PATH", str(path_dir))
        result = resolve_binary(llama_bench=bp)
        assert result.binary_path == bp
        assert result.source == "explicit"

    def test_install_dir_searched(self, tmp_path: Path) -> None:
        install = tmp_path / "install"
        install.mkdir()
        bp = install / "llama-bench"
        bp.write_text("data")
        bp.chmod(0o755)
        result = resolve_binary(install_dir=install)
        assert result.binary_path == bp
        assert result.source == "install_dir"

    def test_install_dir_non_executable_falls_to_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install = tmp_path / "install"
        install.mkdir()
        non_exec = install / "llama-bench"
        non_exec.write_text("data")
        non_exec.chmod(0o644)
        path_dir = tmp_path / "pathdir"
        path_dir.mkdir()
        bp_path = path_dir / "llama-bench"
        bp_path.write_text("data")
        bp_path.chmod(0o755)
        monkeypatch.setenv("PATH", str(path_dir))
        result = resolve_binary(install_dir=install)
        assert result.binary_path == bp_path
        assert result.source == "path"

    def test_install_dir_absent_falls_to_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install = tmp_path / "install"
        install.mkdir()
        path_dir = tmp_path / "pathdir"
        path_dir.mkdir()
        bp_path = path_dir / "llama-bench"
        bp_path.write_text("data")
        bp_path.chmod(0o755)
        monkeypatch.setenv("PATH", str(path_dir))
        result = resolve_binary(install_dir=install)
        assert result.binary_path == bp_path
        assert result.source == "path"

    def test_install_dir_absent_no_path_fallback_raises(self, tmp_path: Path) -> None:
        install = tmp_path / "install"
        install.mkdir()
        with pytest.raises(ToolchainError, match="llama-bench not found"):
            resolve_binary(install_dir=install)

    def test_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path_dir = tmp_path / "bindir"
        path_dir.mkdir()
        bp = path_dir / "llama-bench"
        bp.write_text("data")
        bp.chmod(0o755)
        monkeypatch.setenv("PATH", str(path_dir))
        result = resolve_binary()
        assert result.binary_path == bp
        assert result.source == "path"

    def test_path_non_executable_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path_dir = tmp_path / "bindir"
        path_dir.mkdir()
        non_exec = path_dir / "llama-bench"
        non_exec.write_text("data")
        non_exec.chmod(0o644)
        monkeypatch.setenv("PATH", str(path_dir))
        with pytest.raises(ToolchainError, match="llama-bench not found"):
            resolve_binary()

    def test_path_directory_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path_dir = tmp_path / "bindir"
        path_dir.mkdir()
        d = path_dir / "llama-bench"
        d.mkdir()
        monkeypatch.setenv("PATH", str(path_dir))
        with pytest.raises(ToolchainError, match="llama-bench not found"):
            resolve_binary()

    def test_path_lookup_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(ToolchainError, match="llama-bench not found"):
            resolve_binary()

    def test_install_dir_beats_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        install = tmp_path / "install"
        install.mkdir()
        bp_install = install / "llama-bench"
        bp_install.write_text("data")
        bp_install.chmod(0o755)
        path_dir = tmp_path / "pathdir"
        path_dir.mkdir()
        (path_dir / "llama-bench").touch()
        (path_dir / "llama-bench").chmod(0o755)
        monkeypatch.setenv("PATH", str(path_dir))
        result = resolve_binary(install_dir=install)
        assert result.binary_path == bp_install
        assert result.source == "install_dir"


class TestInspectToolchain:
    def test_success(self, valid_version_output: str) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        version_argv = (str(bp), "--version")
        help_argv = (str(bp), "--help")
        runner.enqueue(
            version_argv,
            ProcessResult(returncode=0, stdout=valid_version_output, stderr=""),
        )
        runner.enqueue(
            help_argv,
            ProcessResult(returncode=0, stdout="usage: llama-bench", stderr=""),
        )

        fp = inspect_toolchain(runner, bp)
        assert fp.binary_path == bp
        assert fp.version == "1234"
        assert fp.supported_flags == (
            "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
            "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
        )
        assert fp.device_capabilities == ("cpu", "metal")
        assert fp.supported_output_formats == ("json", "jsonl")

    def test_version_failure(self) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=1, stdout="", stderr="error"),
        )
        with pytest.raises(ToolchainError, match="version exited with code 1"):
            inspect_toolchain(runner, bp)

    def test_version_empty_output(self) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=0, stdout="", stderr=""),
        )
        with pytest.raises(ToolchainError, match="no output"):
            inspect_toolchain(runner, bp)

    def test_version_no_version_line(self) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=0, stdout="backends: cpu\n"
                                     "supported_output_formats: json\n", stderr=""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
        )
        with pytest.raises(ToolchainError, match="version:' line"):
            inspect_toolchain(runner, bp)

    def test_help_failure(self, valid_version_output: str) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=0, stdout=valid_version_output, stderr=""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(returncode=1, stdout="", stderr="help error"),
        )
        with pytest.raises(ToolchainError, match="help exited with code 1"):
            inspect_toolchain(runner, bp)

    def test_missing_backends_raises(self) -> None:
        output = "version: 1234\nsupported_output_formats: json\n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="backends:' line"):
            inspect_toolchain(runner, bp)

    def test_help_empty_output(self, valid_version_output: str) -> None:
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=0, stdout=valid_version_output, stderr=""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(returncode=0, stdout="", stderr=""),
        )
        with pytest.raises(ToolchainError, match="no output"):
            inspect_toolchain(runner, bp)

    def test_missing_supported_flags_raises(self) -> None:
        output = "version: 1234\nbackends: cpu\nsupported_output_formats: json\n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="supported_flags:' line"):
            inspect_toolchain(runner, bp)

    def test_empty_supported_flags_raises(self) -> None:
        output = "version: 1234\nbackends: cpu\nsupported_flags: \nsupported_output_formats: json\n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="Empty"):
            inspect_toolchain(runner, bp)

    def test_required_flags_missing_rejected(self) -> None:
        output = (
            "version: 1234\nbackends: cpu\n"
            "supported_flags: -fa,-ctk\n"
            "supported_output_formats: json\n"
        )
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="missing required supported flags"):
            inspect_toolchain(runner, bp)

    def test_json_output_format_required(self) -> None:
        output = (
            "version: 1234\nbackends: cpu\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
            "supported_output_formats: csv\n"
        )
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="does not support.*json"):
            inspect_toolchain(runner, bp)

    def test_missing_output_formats_raises(self) -> None:
        output = "version: 1234\nbackends: cpu\n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="supported_output_formats:' line"):
            inspect_toolchain(runner, bp)

    def test_empty_backends_raises(self) -> None:
        output = "version: 1234\nbackends: \nsupported_output_formats: json\n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="Empty"):
            inspect_toolchain(runner, bp)

    def test_empty_output_formats_raises(self) -> None:
        output = "version: 1234\nbackends: cpu\nsupported_output_formats: \n"
        runner = FakeRunner()
        bp = Path("/usr/local/bin/llama-bench")
        runner.enqueue((str(bp), "--version"), ProcessResult(0, output, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage", ""))
        with pytest.raises(ToolchainError, match="Empty"):
            inspect_toolchain(runner, bp)


class TestParseVersion:
    def test_happy_path(self) -> None:
        assert _parse_version("version: 1234\n") == "1234"

    def test_no_match(self) -> None:
        with pytest.raises(ToolchainError, match="version:' line"):
            _parse_version("some text\n")


class TestParseBackends:
    def test_single(self) -> None:
        assert _parse_backends("backends: cuda\n") == ["cuda"]

    def test_multiple(self) -> None:
        assert _parse_backends("backends: cpu,metal,cuda\n") == ["cpu", "metal", "cuda"]

    def test_empty_raises(self) -> None:
        with pytest.raises(ToolchainError, match="Empty"):
            _parse_backends("backends: \n")

    def test_missing_raises(self) -> None:
        with pytest.raises(ToolchainError, match="backends:' line"):
            _parse_backends("version: 1234\n")


class TestParseOutputFormats:
    def test_multiple(self) -> None:
        assert _parse_output_formats("supported_output_formats: json,jsonl,csv\n") == [
            "json", "jsonl", "csv",
        ]

    def test_empty_raises(self) -> None:
        with pytest.raises(ToolchainError, match="Empty"):
            _parse_output_formats("supported_output_formats: \n")

    def test_missing_raises(self) -> None:
        with pytest.raises(ToolchainError, match="supported_output_formats:' line"):
            _parse_output_formats("version: 1234\n")


class TestParseSupportedFlags:
    def test_present(self) -> None:
        assert _parse_supported_flags("supported_flags: -fa,-ctk\n") == ["-fa", "-ctk"]

    def test_empty_raises(self) -> None:
        with pytest.raises(ToolchainError, match="Empty"):
            _parse_supported_flags("supported_flags: \n")

    def test_missing_raises(self) -> None:
        with pytest.raises(ToolchainError, match="supported_flags:' line"):
            _parse_supported_flags("version: 1234\n")
