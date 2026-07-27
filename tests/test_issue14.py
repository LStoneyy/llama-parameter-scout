from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# -- helpers -----------------------------------------------------------------


def _make_fake_gguf(path: Path) -> None:
    """Build a minimal valid GGUF v3 file with required metadata."""
    import struct
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", 3))
    for k, v in [("general.architecture", "llama")]:
        enc = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(enc)))
        buf.extend(enc)
        buf.extend(struct.pack("<I", 8))
        v_enc = v.encode("utf-8")
        buf.extend(struct.pack("<Q", len(v_enc)))
        buf.extend(v_enc)
    for k, vt, val in [
        ("llama.context_length", 4, 4096),
        ("general.file_type", 4, 17),
    ]:
        enc = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(enc)))
        buf.extend(enc)
        buf.extend(struct.pack("<I", vt))
        buf.extend(struct.pack("<i", val))
    path.write_bytes(bytes(buf))


def _make_fake_bench_script(path: Path) -> Path:
    """Create a fake llama-bench executable, return the actual executable Path.

    POSIX: shebang script at *path* with executable bit.
    Windows: Python helper at ``path.py`` and a ``.cmd`` wrapper invoking an
    absolute ``sys.executable`` / helper path that forwards ``%*``.
    """
    content = (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "if '--version' in sys.argv:\n"
        "  print('version: test123')\n"
        "  print('backends: cpu')\n"
        "  print('supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r')\n"
        "  print('supported_output_formats: json')\n"
        "elif '--help' in sys.argv:\n"
        "  print('usage: llama-bench')\n"
        "else:\n"
        "  print(json.dumps([{'samples_ts': [10.0, 11.0, 12.0]}]))\n"
    )
    if sys.platform == "win32":
        helper = path.with_suffix(".py")
        helper.write_text(content)
        cmd = path.with_suffix(".cmd")
        cmd.write_text(f'@"{sys.executable}" "{helper}" %*\r\n')
        return cmd
    path.write_text(content)
    path.chmod(0o755)
    return path


def _clean_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with PYTHONPATH and PYTHONHOME removed."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


# -- network guard fixture ---------------------------------------------------

@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch socket.connect/connect_ex/bind/create_connection to reject
    non-loopback addresses.

    Teardown is automatic via monkeypatch undo.
    """
    import socket
    from typing import Any

    _real_connect = socket.socket.connect
    _real_connect_ex = socket.socket.connect_ex
    _real_bind = socket.socket.bind
    _real_create_connection = socket.create_connection

    def _is_loopback(host: str) -> bool:
        return host in ("127.0.0.1", "::1", "localhost")

    def _addr_str(addr: tuple[Any, ...] | str | bytes) -> str:
        if isinstance(addr, bytes):
            return addr.decode("utf-8", errors="replace")
        return str(addr)

    def _guarded_connect(
        self: socket.socket, addr: tuple[Any, ...] | str | bytes,
        *a: object, **kw: object,
    ) -> None:
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if isinstance(host, str) and _is_loopback(host):
            _real_connect(self, addr, *a, **kw)
        else:
            raise RuntimeError(
                "Blocked outbound connection to {} during network-safety test".format(
                    _addr_str(addr),
                )
            )

    def _guarded_connect_ex(
        self: socket.socket, addr: tuple[Any, ...] | str | bytes,
        *a: object, **kw: object,
    ) -> int:
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if isinstance(host, str) and _is_loopback(host):
            return _real_connect_ex(self, addr, *a, **kw)
        return -1

    def _guarded_bind(
        self: socket.socket, addr: tuple[Any, ...] | str | bytes,
        *a: object, **kw: object,
    ) -> None:
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if isinstance(host, str) and _is_loopback(host):
            _real_bind(self, addr, *a, **kw)
        else:
            raise RuntimeError(
                "Blocked bind to {} during network-safety test (only loopback allowed)".format(
                    _addr_str(addr),
                )
            )

    def _guarded_create_connection(
        address: Any,
        timeout: float | None = None,
        source_address: Any = None,
        *,
        all_errors: bool = False,
    ) -> socket.socket:
        host: str
        if isinstance(address, tuple):
            host = str(address[0]) if address else ""
        else:
            host = str(address)
        if _is_loopback(host):
            return _real_create_connection(address, timeout, source_address, all_errors=all_errors)
        raise RuntimeError(
            "Blocked outbound socket.create_connection to {} "
            "during network-safety test".format(_addr_str(address))
        )

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "bind", _guarded_bind)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)


# -- smoke venv fixture (module-scoped, builds once) -------------------------

@pytest.fixture(scope="module")
def _smoke_venv(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Build wheel once and install into an isolated venv for all smoke tests."""
    root = Path(__file__).resolve().parent.parent
    dist_dir = tmp_path_factory.mktemp("dist")
    venv_dir = tmp_path_factory.mktemp("venv")

    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        capture_output=True, text=True, cwd=root, check=True,
        env=_clean_env(),
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) >= 1, "No wheel produced by build"

    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True, text=True, check=True,
        env=_clean_env(),
    )

    pip = str(venv_dir / "bin" / "pip") if sys.platform != "win32" else str(
        venv_dir / "Scripts" / "pip.exe"
    )
    subprocess.run(
        [pip, "install", str(wheels[0])],
        capture_output=True, text=True, check=True,
        env=_clean_env(),
    )

    python = str(venv_dir / "bin" / "python") if sys.platform != "win32" else str(
        venv_dir / "Scripts" / "python.exe"
    )
    entry = str(venv_dir / "bin" / "llama-parameter-scout") if sys.platform != "win32" else str(
        venv_dir / "Scripts" / "llama-parameter-scout.exe"
    )

    return {"python": python, "entry": entry, "venv_dir": str(venv_dir)}


# -- TestPackageImport ---------------------------------------------------------

class TestPackageImport:
    """Verify the package imports correctly and exposes expected names."""

    def test_import_cli(self) -> None:
        from llama_parameter_scout import cli
        assert hasattr(cli, "main")

    def test_import_application(self) -> None:
        from llama_parameter_scout import application
        assert hasattr(application, "Application")

    def test_import_models(self) -> None:
        from llama_parameter_scout import models
        assert hasattr(models, "SingleRunRequest")
        assert hasattr(models, "CandidateConfig")
        assert hasattr(models, "SessionSpecification")

    def test_import_tui(self) -> None:
        from llama_parameter_scout import tui
        assert hasattr(tui, "ScoutTuiApp")

    def test_import_runner(self) -> None:
        from llama_parameter_scout import runner
        assert hasattr(runner, "FakeRunner")
        assert hasattr(runner, "ProductionRunner")

    def test_import_reporter(self) -> None:
        from llama_parameter_scout import reporter
        assert hasattr(reporter, "render_export_json")
        assert hasattr(reporter, "render_export_markdown")

    def test_import_search(self) -> None:
        from llama_parameter_scout import search
        assert hasattr(search, "screen")
        assert hasattr(search, "QUICK_PROFILE")


# -- TestEntryPoint ------------------------------------------------------------

class TestEntryPoint:
    """Verify the console entry point is registered and invocable."""

    def test_entry_point_in_metadata(self) -> None:
        dist = importlib.metadata.distribution("llama-parameter-scout")
        eps = dist.entry_points
        console_scripts = [ep for ep in eps if ep.group == "console_scripts"]
        assert any(
            ep.name == "llama-parameter-scout" and
            ep.value == "llama_parameter_scout.cli:main"
            for ep in console_scripts
        ), f"llama-parameter-scout entry point not found in {console_scripts}"

    def test_cli_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "llama_parameter_scout.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()

    def test_cli_tui_flag_accepted(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "llama_parameter_scout.cli", "--tui", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()

    def test_cli_list_sessions_requires_journal(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "llama_parameter_scout.cli", "--list-sessions"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr


# -- TestNoOutboundNetwork -----------------------------------------------------

class TestNoOutboundNetwork:
    """Verify the package does not open outbound network connections at rest.

    These tests monkeypatch socket.connect/connect_ex/bind to block
    non-loopback addresses. The monkeypatch fixture ensures the originals
    are restored after each test (no leaking).
    """

    @pytest.mark.network
    def test_no_network_during_import(self, block_network: None) -> None:
        import llama_parameter_scout as lps
        _ = lps.__name__
        from llama_parameter_scout import (
            application,
            cli,
            models,
            reporter,
            runner,
            search,
            toolchain,
            topology,
        )
        _ = application, cli, models, reporter, runner, search, toolchain, topology

    @pytest.mark.network
    def test_no_network_during_cli_help(self, block_network: None) -> None:
        from llama_parameter_scout.cli import _parse_args

        try:
            _parse_args(["--help"])
        except SystemExit:
            pass

    @pytest.mark.network
    def test_no_network_during_tui_import(self, block_network: None) -> None:
        from llama_parameter_scout.tui import ScoutTuiApp
        _ = ScoutTuiApp

    @pytest.mark.network
    def test_no_network_during_fake_adapter_flow(
        self, tmp_path: Path, block_network: None,
    ) -> None:
        """Run a full fake-adapter pipeline under socket block."""
        import json

        from llama_parameter_scout.application import Application
        from llama_parameter_scout.models import (
            CandidateConfig,
            NctxComposition,
            SingleRunRequest,
        )
        from llama_parameter_scout.runner import FakeRunner, ProcessResult
        from tests.conftest import _build_minimal_gguf

        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho ok")
        bp.chmod(0o755)

        mp = tmp_path / "model.gguf"
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})

        runner = FakeRunner()
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=0, stdout=(
                "version: test\nbackends: cpu\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
                "supported_output_formats: json\n"
            ), stderr=""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
        )

        jp = tmp_path / "journal.json"

        n_ctx = NctxComposition(total=64, prompt=16, generation=32, depth=16)
        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )

        request = SingleRunRequest(
            binary_path=bp, model_path=mp, journal_path=jp,
            n_ctx=n_ctx, candidate=candidate,
        )

        app = Application(runner)

        warmup_argv = (
            str(bp), "-m", str(mp), "-p", "16", "-n", "32", "-d", "16",
            "-t", "4", "-b", "512", "-ub", "512", "-ngl", "0",
            "-fa", "off", "-ctk", "f16", "-ctv", "f16",
            "-o", "json", "-r", "1",
        )
        measure_argv = (
            str(bp), "-m", str(mp), "-p", "16", "-n", "32", "-d", "16",
            "-t", "4", "-b", "512", "-ub", "512", "-ngl", "0",
            "-fa", "off", "-ctk", "f16", "-ctv", "f16",
            "-o", "json", "-r", "3",
        )

        runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=json.dumps([{
                "samples_ts": [10.0, 11.0, 12.0],
            }]), stderr=""),
        )
        runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=json.dumps([{
                "samples_ts": [45.5, 44.8, 45.3],
            }]), stderr=""),
        )

        obs = app.run(request)
        assert obs is not None
        assert obs.median_tps == pytest.approx(45.3, rel=1e-6)


# -- TestCliRepresentativeFlows ------------------------------------------------

class TestCliRepresentativeFlows:
    """Verify CLI flows using fake subprocess binaries."""

    def _make_llama_bench_script(self, path: Path) -> Path:
        """Python-based fake llama-bench for cross-platform use."""
        return _make_fake_bench_script(path)

    def _make_gguf(self, path: Path) -> None:
        _make_fake_gguf(path)

    def test_cli_error_on_missing_binary(self, tmp_path: Path) -> None:
        mp = tmp_path / "model.gguf"
        self._make_gguf(mp)
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_parameter_scout.cli",
                "--llama-bench", "/nonexistent",
                "--model-path", str(mp),
                "--journal-path", str(tmp_path / "j.json"),
                "--n-ctx", "2048", "--n-prompt", "512",
                "--n-generation", "1024", "--n-depth", "512",
                "--threads", "4", "--batch-size", "512",
                "--ubatch-size", "512", "--n-gpu-layers", "0",
                "--cpu-only", "--yes",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr

    def test_cli_success_flow(self, tmp_path: Path) -> None:
        bp = self._make_llama_bench_script(tmp_path / "llama-bench")
        mp = tmp_path / "model.gguf"
        self._make_gguf(mp)
        jp = tmp_path / "journal.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_parameter_scout.cli",
                "--llama-bench", str(bp),
                "--model-path", str(mp),
                "--journal-path", str(jp),
                "--n-ctx", "2048", "--n-prompt", "512",
                "--n-generation", "1024", "--n-depth", "512",
                "--threads", "4", "--batch-size", "512",
                "--ubatch-size", "512", "--n-gpu-layers", "0",
                "--cpu-only", "--yes",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Toolchain (from explicit):" in result.stdout
        assert "Session Specification:" in result.stdout
        assert "Warmup complete" in result.stdout
        assert "Median throughput:" in result.stdout
        assert "Journal:" in result.stdout
        assert jp.exists()

    def test_cli_list_sessions(self, tmp_path: Path) -> None:
        jp = tmp_path / "journal.json"
        jp.write_text('{"schema_version": 5, "events": []}')
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_parameter_scout.cli",
                "--list-sessions", "--journal-path", str(jp),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "No sessions found." in result.stdout or "Known sessions:" in result.stdout

    def test_cli_list_sessions_with_known(self, tmp_path: Path) -> None:
        jp = tmp_path / "journal.json"
        jp.write_text(
            '{"schema_version": 5, "events": ['
            '{"event_type": "session_start", "session_id": "abc123", '
            '"fingerprint_components": {"schema_version": 5}}'
            "]}",
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_parameter_scout.cli",
                "--list-sessions", "--journal-path", str(jp),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "abc123" in result.stdout


# -- TestTuiRepresentativeFlow -------------------------------------------------

class TestTuiRepresentativeFlow:
    """Verify TUI imports and can be constructed (no real display)."""

    def test_tui_app_constructable(self) -> None:
        from llama_parameter_scout.tui import ScoutTuiApp
        app = ScoutTuiApp()
        assert app.title == "llama-parameter-scout"


# -- TestCpuOnlySmoke ----------------------------------------------------------

class TestCpuOnlySmoke:
    """CPU-only smoke that exercises Application with FakeRunner.

    These tests represent the deterministic no-real-hardware smoke path.
    They run as part of the default test suite (not marked smoke) so every
    CI run exercises them. Smoke-only gate is enforced by the CI workflow
    that explicitly runs the smoke marker.
    """

    def test_application_run_with_fake_runner(self, tmp_path: Path) -> None:
        import json

        from llama_parameter_scout.application import Application
        from llama_parameter_scout.models import (
            CandidateConfig,
            NctxComposition,
            SingleRunRequest,
        )
        from llama_parameter_scout.runner import FakeRunner, ProcessResult
        from tests.conftest import _build_minimal_gguf

        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho ok")
        bp.chmod(0o755)

        mp = tmp_path / "model.gguf"
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})
        jp = tmp_path / "journal.json"

        runner = FakeRunner()
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(0, (
                "version: test\nbackends: cpu\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
                "supported_output_formats: json\n"
            ), ""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(0, "usage: llama-bench\n", ""),
        )
        runner.enqueue(
            (str(bp), "-m", str(mp), "-p", "512", "-n", "1024", "-d", "512",
             "-t", "4", "-b", "512", "-ub", "512", "-ngl", "0",
             "-fa", "off", "-ctk", "f16", "-ctv", "f16",
             "-o", "json", "-r", "1"),
            ProcessResult(0, json.dumps([{"samples_ts": [10.0, 11.0, 12.0]}]), ""),
        )
        runner.enqueue(
            (str(bp), "-m", str(mp), "-p", "512", "-n", "1024", "-d", "512",
             "-t", "4", "-b", "512", "-ub", "512", "-ngl", "0",
             "-fa", "off", "-ctk", "f16", "-ctv", "f16",
             "-o", "json", "-r", "3"),
            ProcessResult(0, json.dumps([{"samples_ts": [45.5, 44.8, 45.3]}]), ""),
        )

        app = Application(runner)
        request = SingleRunRequest(
            binary_path=bp, model_path=mp, journal_path=jp,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )
        obs = app.run(request)
        assert obs is not None
        assert obs.median_tps == pytest.approx(45.3, rel=1e-6)
        assert obs.candidate.n_gpu_layers == 0

    def test_application_screen_candidates_smoke(self, tmp_path: Path) -> None:
        from collections.abc import Sequence

        from llama_parameter_scout.application import Application
        from llama_parameter_scout.models import (
            CandidateConfig,
            NctxComposition,
            SingleRunRequest,
        )
        from llama_parameter_scout.runner import FakeRunner, ProcessResult
        from llama_parameter_scout.search import QUICK_PROFILE
        from tests.conftest import _build_minimal_gguf

        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho ok")
        bp.chmod(0o755)

        mp = tmp_path / "model.gguf"
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})
        jp = tmp_path / "journal.json"

        runner = FakeRunner()
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(0, (
                "version: test\nbackends: cpu\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
                "supported_output_formats: json\n"
            ), ""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(0, "usage: llama-bench\n", ""),
        )

        request = SingleRunRequest(
            binary_path=bp, model_path=mp, journal_path=jp,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )

        adapter_calls: list[list[str]] = []

        class RecordingAdapter:
            def launch(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None,
                cancellation: threading.Event | None = None,
                memory_pressure_fn: object = None,
                memory_pressure_backoff_seconds: float = 1.0,
                memory_pressure_max_backoffs: int = 3,
            ) -> ProcessResult:
                adapter_calls.append(list(argv))
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=RecordingAdapter())
        report = app.screen_candidates(request, QUICK_PROFILE)
        assert report.baseline_disposition is not None
        assert adapter_calls, "Adapter should have been called"


# -- TestPackageSmoke (release gate: runs from installed wheel) ----------------

class TestPackageSmoke:
    """Verify the built wheel installs cleanly into an isolated venv.

    These tests are marked 'smoke' and run as a CI release gate. They
    build a wheel once (module-scoped fixture), install it into an
    isolated venv, and verify:

    - Package import resolves from the installed environment.
    - Console entry point exists and is invocable.
    - A representative fake-adapter Application flow works.
    - A TUI import and basic lifecycle work.

    Every assertion uses the installed venv's Python/entry point, never
    the checkout tree.
    """

    # ========================================================================
    # Installed-package import
    # ========================================================================

    @pytest.mark.smoke
    def test_installed_package_import(self, tmp_path: Path, _smoke_venv: dict[str, str]) -> None:
        script = tmp_path / "_test_import.py"
        script.write_text(
            "import sys, os\n"
            "import llama_parameter_scout\n"
            "venv = os.environ['_TEST_VENV_DIR']\n"
            "f = llama_parameter_scout.__file__\n"
            "assert f.startswith(venv), f'{f} not in {venv}'\n"
            "from llama_parameter_scout import cli, application\n"
            "from llama_parameter_scout import models, tui, runner, reporter, search\n"
            "from llama_parameter_scout.models import SingleRunRequest\n"
            "print('ok')\n"
        )
        env = _clean_env()
        env['_TEST_VENV_DIR'] = _smoke_venv['venv_dir']
        result = subprocess.run(
            [_smoke_venv["python"], str(script)],
            capture_output=True, text=True,
            cwd=tmp_path,
            env=env,
        )
        assert result.returncode == 0, f"Import failed:\n{result.stderr}"
        assert "ok" in result.stdout

    # ========================================================================
    # Installed console entry point
    # ========================================================================

    @pytest.mark.smoke
    def test_installed_entry_point_exists(self, _smoke_venv: dict[str, str]) -> None:
        entry = Path(_smoke_venv["entry"])
        assert entry.exists(), f"Entry point not found at {entry}"

    @pytest.mark.smoke
    def test_installed_entry_point_help(self, _smoke_venv: dict[str, str]) -> None:
        result = subprocess.run(
            [_smoke_venv["entry"], "--help"],
            capture_output=True, text=True,
            cwd=Path(_smoke_venv["venv_dir"]),
            env=_clean_env(),
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()

    @pytest.mark.smoke
    def test_installed_entry_point_cli_flow(
        self, tmp_path: Path, _smoke_venv: dict[str, str],
    ) -> None:
        bp = _make_fake_bench_script(tmp_path / "llama-bench")
        mp = tmp_path / "model.gguf"
        _make_fake_gguf(mp)
        jp = tmp_path / "journal.json"
        result = subprocess.run(
            [
                _smoke_venv["entry"],
                "--llama-bench", str(bp),
                "--model-path", str(mp),
                "--journal-path", str(jp),
                "--n-ctx", "2048", "--n-prompt", "512",
                "--n-generation", "1024", "--n-depth", "512",
                "--threads", "4", "--batch-size", "512",
                "--ubatch-size", "512", "--n-gpu-layers", "0",
                "--cpu-only", "--yes",
            ],
            capture_output=True, text=True,
            cwd=tmp_path,
            env=_clean_env(),
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "Warmup complete" in result.stdout

    # ========================================================================
    # Installed-package fake-adapter Application flow
    # ========================================================================

    @pytest.mark.smoke
    def test_installed_fake_adapter_flow(self, tmp_path: Path, _smoke_venv: dict[str, str]) -> None:
        script = tmp_path / "_test_fake_adapter.py"
        script.write_text((
            "import json, sys, struct\n"
            "from pathlib import Path\n"
            "\n"
            "def _make_gguf(path):\n"
            "    buf = bytearray(b'GGUF')\n"
            "    buf += struct.pack('<I', 3) + struct.pack('<Q', 0)\n"
            "    buf += struct.pack('<Q', 3)\n"
            "    for k, vt, val in [('general.architecture', 8, 'llama'),\n"
            "                       ('llama.context_length', 4, 4096),\n"
            "                       ('general.file_type', 4, 17)]:\n"
            "        enc = k.encode(); buf += struct.pack('<Q', len(enc)) + enc\n"
            "        buf += struct.pack('<I', vt)\n"
            "        if isinstance(val, str):\n"
            "            ve = val.encode(); buf += struct.pack('<Q', len(ve)) + ve\n"
            "        else:\n"
            "            buf += struct.pack('<i', val)\n"
            "    path.write_bytes(bytes(buf))\n"
            "\n"
"bp = Path(sys.argv[1])\n"
"bp.write_text("
"'#!/usr/bin/env python3\\n"
"import json,sys\\n"
"print(json.dumps([{\\'samples_ts\\':[1.0,2.0,3.0]}]))\\n')\n"
"bp.chmod(0o755)\n"
"mp = Path(sys.argv[2] + '.gguf')\n"
"_make_gguf(mp)\n"
"jp = Path(sys.argv[3])\n"
            "from llama_parameter_scout.application import Application\n"
            "from llama_parameter_scout.models import (CandidateConfig,\n"
            "    NctxComposition, SingleRunRequest)\n"
            "from llama_parameter_scout.runner import FakeRunner, ProcessResult\n"
            "runner = FakeRunner()\n"
"runner.enqueue((str(bp), '--version'),\n"
"    ProcessResult(0, 'version: test\\nbackends: cpu\\n'\n"
"    'supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\\n'\n"
"    'supported_output_formats: json\\n', ''))\n"
"runner.enqueue((str(bp), '--help'),\n"
"    ProcessResult(0, 'usage: llama-bench\\n', ''))\n"
            "runner.enqueue((str(bp), '-m', str(mp), '-p', '16', '-n', '32',\n"
            "    '-d', '16', '-t', '4', '-b', '512', '-ub', '512', '-ngl', '0',\n"
            "    '-fa', 'off', '-ctk', 'f16', '-ctv', 'f16', '-o', 'json', '-r', '1'),\n"
            "    ProcessResult(0, json.dumps([{'samples_ts': [10.0, 11.0, 12.0]}]), ''))\n"
            "runner.enqueue((str(bp), '-m', str(mp), '-p', '16', '-n', '32',\n"
            "    '-d', '16', '-t', '4', '-b', '512', '-ub', '512', '-ngl', '0',\n"
            "    '-fa', 'off', '-ctk', 'f16', '-ctv', 'f16', '-o', 'json', '-r', '3'),\n"
            "    ProcessResult(0, json.dumps([{'samples_ts':[45.5,44.8,45.3]}]), ''))\n"
            "app = Application(runner)\n"
            "request = SingleRunRequest(binary_path=bp, model_path=mp,\n"
            "    journal_path=jp,\n"
            "    n_ctx=NctxComposition(total=64, prompt=16, generation=32, depth=16),\n"
            "    candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,\n"
            "        n_gpu_layers=0, flash_attn=False,\n"
            "        cache_type_k='f16', cache_type_v='f16'))\n"
            "obs = app.run(request)\n"
            "assert obs is not None\n"
            "assert obs.median_tps == 45.3\n"
            "print(f'median_tps={obs.median_tps}')"
        ))
        result = subprocess.run(
            [
                _smoke_venv["python"], str(script),
                str(tmp_path / "bp"), str(tmp_path / "mp"),
                str(tmp_path / "jp"),
            ],
            capture_output=True, text=True,
            cwd=tmp_path,
            env=_clean_env(),
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Fake-adapter flow failed:\n{result.stderr}"
        assert "median_tps=45.3" in result.stdout

    # ========================================================================
    # Installed-package TUI path
    # ========================================================================

    @pytest.mark.smoke
    def test_installed_tui_import_and_lifecycle(
        self, tmp_path: Path, _smoke_venv: dict[str, str],
    ) -> None:
        script = tmp_path / "_test_tui.py"
        script.write_text(
            "import asyncio\n"
            "from llama_parameter_scout.tui import ScoutTuiApp\n"
            "async def main():\n"
            "    app = ScoutTuiApp()\n"
            "    async with app.run_test() as pilot:\n"
            "        title = app.title\n"
            "        print(f'title={title}')\n"
            "asyncio.run(main())\n"
        )
        result = subprocess.run(
            [_smoke_venv["python"], str(script)],
            capture_output=True, text=True,
            cwd=tmp_path,
            env=_clean_env(),
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"TUI lifecycle failed:\n{result.stderr}"
        assert "title=llama-parameter-scout" in result.stdout
