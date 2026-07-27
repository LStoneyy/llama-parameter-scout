from __future__ import annotations

import struct
from pathlib import Path

import pytest

from llama_parameter_scout.cli import _parse_args, main


def _make_gguf(path: Path) -> None:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", 3))
    for key, value in [("general.architecture", "llama")]:
        _pack_str(buf, key)
        buf.extend(struct.pack("<I", 8))
        _pack_str(buf, value)
    _pack_str(buf, "llama.context_length")
    buf.extend(struct.pack("<I", 5))
    buf.extend(struct.pack("<i", 4096))
    _pack_str(buf, "general.file_type")
    buf.extend(struct.pack("<I", 5))
    buf.extend(struct.pack("<i", 17))
    path.write_bytes(bytes(buf))


def _pack_str(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _make_version_output() -> str:
    return (
        "version: test123\n"
        "build_commit: aabbcc\n"
        "backends: cpu,metal\n"
        "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
        "supported_output_formats: json,jsonl\n"
    )


class TestParseArgs:
    def test_minimal_args(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        jp = tmp_path / "journal.json"
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "2048",
            "--n-prompt", "512",
            "--n-generation", "1024",
            "--n-depth", "512",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
        ])
        assert args.n_ctx == 2048
        assert args.n_prompt == 512
        assert args.flash_attn is False
        assert args.cache_type_k == "f16"
        assert args.cache_type_v == "f16"
        assert args.binary_path == bp
        assert args.install_dir is None

    def test_install_dir_arg(self, tmp_path: Path) -> None:
        install = tmp_path / "install"
        install.mkdir()
        bp = install / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--install-dir", str(install),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1",
            "--n-prompt", "1",
            "--n-generation", "0",
            "--n-depth", "0",
            "--threads", "1",
            "--batch-size", "1",
            "--ubatch-size", "1",
            "--n-gpu-layers", "0",
            "--cpu-only",
        ])
        assert args.install_dir == install
        assert args.binary_path is None

    def test_llama_bench_overrides_install_dir(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        install = tmp_path / "install"
        install.mkdir()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--install-dir", str(install),
            "--model-path", str(tmp_path / "m.gguf"),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1",
            "--n-prompt", "1",
            "--n-generation", "0",
            "--n-depth", "0",
            "--threads", "1",
            "--batch-size", "1",
            "--ubatch-size", "1",
            "--n-gpu-layers", "0",
            "--cpu-only",
        ])
        assert args.binary_path == bp
        assert args.install_dir == install

    def test_flash_attn_default_false(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1",
            "--n-prompt", "1",
            "--n-generation", "0",
            "--n-depth", "0",
            "--threads", "1",
            "--batch-size", "1",
            "--ubatch-size", "1",
            "--n-gpu-layers", "0",
            "--cpu-only",
        ])
        assert args.flash_attn is False

    def test_flash_attn_flag_sets_true(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1",
            "--n-prompt", "1",
            "--n-generation", "0",
            "--n-depth", "0",
            "--threads", "1",
            "--batch-size", "1",
            "--ubatch-size", "1",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--flash-attn",
        ])
        assert args.flash_attn is True

    def test_cache_types(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1",
            "--n-prompt", "1",
            "--n-generation", "0",
            "--n-depth", "0",
            "--threads", "1",
            "--batch-size", "1",
            "--ubatch-size", "1",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
        ])
        assert args.cache_type_k == "q8_0"
        assert args.cache_type_v == "q4_0"

    def test_cpu_only_flag(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
        ])
        assert args.cpu_only is True

    def test_device_flag(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--device", "2",
        ])
        assert args.device == [2]

    def test_yes_flag(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])
        assert args.yes is True

    def test_budget_flags(self, tmp_path: Path) -> None:
        bp = tmp_path / "llama-bench"
        bp.touch()
        mp = tmp_path / "model.gguf"
        mp.touch()
        args = _parse_args([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(tmp_path / "j.json"),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
            "--budget-ram", "8589934592", "--budget-vram", "1073741824",
        ])
        assert args.budget_ram == 8589934592
        assert args.budget_vram == 1073741824


class TestMain:
    def test_error_prints_to_stderr(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"
        with pytest.raises(SystemExit) as exc:
            main([
                "--llama-bench", "/nonexistent",
                "--model-path", str(mp),
                "--journal-path", str(jp),
                "--n-ctx", "2048",
                "--n-prompt", "512",
                "--n-generation", "1024",
                "--n-depth", "512",
                "--threads", "4",
                "--batch-size", "512",
                "--ubatch-size", "512",
                "--n-gpu-layers", "0",
                "--cpu-only",
                "--yes",
            ])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Disposition: unsupported" in captured.out

    def test_success_output(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu,metal"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json,jsonl"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"

        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "2048",
            "--n-prompt", "512",
            "--n-generation", "1024",
            "--n-depth", "512",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])

        captured = capsys.readouterr()
        assert "Toolchain (from explicit):" in captured.out
        assert "Version: test123" in captured.out
        assert "Flags: -m, -p, -n, -d, -t, -b, -ub, -ngl, -fa, -ctk, -ctv, -o, -r" in captured.out
        assert "Backends: cpu, metal" in captured.out
        assert "Output formats: json, jsonl" in captured.out
        assert "Model:" in captured.out
        assert "Session Specification:" in captured.out
        assert "CPU-only: True" in captured.out
        assert "Resource Budget:" in captured.out
        assert "Context Advisory:" in captured.out
        assert "Fingerprint:" in captured.out
        assert "Warmup complete" in captured.out
        assert "Measurements: 3 repetitions" in captured.out
        assert "Median throughput:" in captured.out
        assert "Journal:" in captured.out

        assert jp.exists()
        import json
        data = json.loads(jp.read_text())
        assert data["version"] == 4

    def test_success_with_install_dir(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        install = tmp_path / "install"
        install.mkdir()
        bp = install / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test456"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"

        main([
            "--install-dir", str(install),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "2048",
            "--n-prompt", "512",
            "--n-generation", "1024",
            "--n-depth", "512",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])

        captured = capsys.readouterr()
        assert "Toolchain (from install_dir):" in captured.out
        assert "Backends: cpu" in captured.out

    def test_supported_flags_persisted_in_journal(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"
        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])
        import json
        data = json.loads(jp.read_text())
        assert data["toolchain"]["supported_flags"] == [
            "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
            "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
        ]

    def test_path_resolution(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bindir = tmp_path / "bindir"
        bindir.mkdir()
        bp = bindir / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: path_test"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        monkeypatch.setenv("PATH", str(bindir))
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"

        main([
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "2048",
            "--n-prompt", "512",
            "--n-generation", "1024",
            "--n-depth", "512",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])

        captured = capsys.readouterr()
        assert "Toolchain (from path):" in captured.out
        assert "Version: path_test" in captured.out

    def test_missing_required_arg(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--llama-bench", "/x"])

    def test_cpu_only_on_host_without_gpu(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"

        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "2048",
            "--n-prompt", "512",
            "--n-generation", "1024",
            "--n-depth", "512",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])
        captured = capsys.readouterr()
        assert "CPU-only: True" in captured.out
        assert "Warmup complete" in captured.out

    def test_session_spec_in_journal(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"
        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])
        import json
        data = json.loads(jp.read_text())
        assert "session_spec" in data
        ss = data["session_spec"]
        assert ss["cpu_only"] is True
        assert ss["selected_device_ids"] == []
        assert ss["n_ctx"]["total"] == 1
        assert ss["acknowledged"] is True
        assert ss["fingerprint"] != ""
        assert "budget" in ss
        assert "advisory" in ss
        assert "topology" in ss
        assert "model" in ss
        assert ss["model"]["architecture"] == "llama"
        assert ss["budget"]["ram_bytes"] > 0
        assert ss["advisory"]["note"] != ""
        assert ss["advisory"]["training_context_status"] == "known"
        assert ss["advisory"]["training_context_tokens"] == 4096

    def test_budget_ram_override(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"
        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "1", "--n-prompt", "1", "--n-generation", "0", "--n-depth", "0",
            "--threads", "1", "--batch-size", "1", "--ubatch-size", "1", "--n-gpu-layers", "0",
            "--cpu-only",
            "--budget-ram", "8589934592",
            "--yes",
        ])
        import json
        data = json.loads(jp.read_text())
        budget = data["session_spec"]["budget"]
        assert budget["ram_bytes"] == 8589934592

    def test_context_above_training_length_warning_output(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        bp = tmp_path / "llama-bench"
        bp.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "version: test123"\n'
            '  echo "backends: cpu"\n'
            '  echo "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"\n'
            '  echo "supported_output_formats: json"\n'
            "elif [ \"$1\" = \"--help\" ]; then\n"
            '  echo "usage: llama-bench"\n'
            "else\n"
            '  echo \'[{"samples_ts":[10.0,11.0,12.0]}]\'\n'
            "fi\n",
        )
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        _make_gguf(mp)
        jp = tmp_path / "journal.json"

        main([
            "--llama-bench", str(bp),
            "--model-path", str(mp),
            "--journal-path", str(jp),
            "--n-ctx", "8192",
            "--n-prompt", "4096",
            "--n-generation", "2048",
            "--n-depth", "2048",
            "--threads", "4",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--n-gpu-layers", "0",
            "--cpu-only",
            "--yes",
        ])
        captured = capsys.readouterr()
        assert "Session Specification:" in captured.out
        assert "Acknowledged: True" in captured.out
        assert "Fingerprint:" in captured.out
