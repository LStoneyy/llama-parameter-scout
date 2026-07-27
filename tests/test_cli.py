from __future__ import annotations

from pathlib import Path

import pytest

from llama_parameter_scout.cli import _parse_args, main


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
        ])
        assert args.n_ctx == 2048
        assert args.n_prompt == 512
        assert args.flash_attn is False
        assert args.cache_type_k == "f16"
        assert args.cache_type_v == "f16"

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
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
        ])
        assert args.cache_type_k == "q8_0"
        assert args.cache_type_v == "q4_0"


class TestMain:
    def test_error_prints_to_stderr(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """CLI catches ScoutError and prints to stderr."""
        mp = tmp_path / "model.gguf"
        mp.touch()
        with pytest.raises(SystemExit) as exc:
            main([
                "--llama-bench", "/nonexistent",
                "--model-path", str(mp),
                "--journal-path", str(tmp_path / "j.json"),
                "--n-ctx", "2048",
                "--n-prompt", "512",
                "--n-generation", "1024",
                "--n-depth", "512",
                "--threads", "4",
                "--batch-size", "512",
                "--ubatch-size", "512",
                "--n-gpu-layers", "0",
            ])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_success_output(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Shell fixture integration: fake binary returns same JSON for both invocations."""
        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho '[{\"samples_ts\":[10.0,11.0,12.0]}]'")
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        mp.write_text("data")
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
        ])

        captured = capsys.readouterr()
        assert "Warmup complete" in captured.out
        assert "Measurements: 3 repetitions" in captured.out
        assert "Median throughput:" in captured.out
        assert "Dispersion" in captured.out
        assert "Journal:" in captured.out

        assert jp.exists()
        import json
        data = json.loads(jp.read_text())
        assert data["version"] == 1

    def test_missing_required_arg(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--llama-bench", "/x"])
