from __future__ import annotations

import pytest

from llama_parameter_scout.runner import FakeRunner, ProcessResult


def test_fake_runner_enqueue_and_run() -> None:
    runner = FakeRunner()
    argv = ("llama-bench", "-m", "model")
    result = ProcessResult(returncode=0, stdout="ok", stderr="")
    runner.enqueue(argv, result)
    assert runner.run(argv) == result
    assert runner.calls == [argv]


def test_fake_runner_fails_on_unexpected_argv() -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError, match="unexpected argv"):
        runner.run(("unexpected",))
