from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    def run(self, argv: Sequence[str]) -> ProcessResult:
        ...


class ProductionRunner:
    def run(self, argv: Sequence[str]) -> ProcessResult:
        result = subprocess.run(
            [str(a) for a in argv],
            capture_output=True,
            text=True,
        )
        return ProcessResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


@dataclass
class FakeRunner:
    _results: dict[tuple[str, ...], ProcessResult] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def enqueue(self, argv: Sequence[str], result: ProcessResult) -> None:
        self._results[tuple(argv)] = result

    def run(self, argv: Sequence[str]) -> ProcessResult:
        key = tuple(argv)
        self.calls.append(key)
        if key not in self._results:
            raise ValueError(f"unexpected argv: {' '.join(argv)}")
        return self._results[key]
