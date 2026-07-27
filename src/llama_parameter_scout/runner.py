from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Protocol

from .models import ScoutError


class ProcessStartError(ScoutError):
    """The subprocess could not be started (Popen construction failed)."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    termination: str | None = None


class ProcessRunner(Protocol):
    def run(self, argv: Sequence[str]) -> ProcessResult:
        """Run the command and return a ProcessResult.

        The implementation must not raise FileNotFoundError or similar
        OS-level exceptions for missing executables; those should be
        returned as a non-zero returncode ProcessResult.
        """


class ProductionRunner:
    def run(self, argv: Sequence[str]) -> ProcessResult:
        try:
            result = subprocess.run(
                [str(a) for a in argv],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            return ProcessResult(
                returncode=127,
                stdout="",
                stderr=str(e),
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


# ── Process Adapter for launched candidates ────────────────────


class ProcessAdapter(Protocol):
    def launch(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        cancellation: Event | None = None,
        memory_pressure_fn: Callable[[], bool] | None = None,
        memory_pressure_backoff_seconds: float = 1.0,
        memory_pressure_max_backoffs: int = 3,
    ) -> ProcessResult:
        ...


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.returncode is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        except (OSError, ProcessLookupError):
            pass


class ProductionProcessAdapter:
    def launch(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        cancellation: Event | None = None,
        memory_pressure_fn: Callable[[], bool] | None = None,
        memory_pressure_backoff_seconds: float = 1.0,
        memory_pressure_max_backoffs: int = 3,
    ) -> ProcessResult:
        base_env = os.environ.copy()
        if env:
            base_env.update(env)

        start_new_session = False
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            start_new_session = True

        try:
            proc = subprocess.Popen(
                [str(a) for a in argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=base_env,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
        except OSError as e:
            raise ProcessStartError(str(e)) from e

        stdout_data: list[str] = []
        stderr_data: list[str] = []

        def _read_stream(stream: Any, dest: list[str]) -> None:
            if stream is None:
                return
            try:
                data = stream.read()
                dest.append(data)
            except OSError:
                pass

        t_out = threading.Thread(
            target=_read_stream, args=(proc.stdout, stdout_data), daemon=True
        )
        t_err = threading.Thread(
            target=_read_stream, args=(proc.stderr, stderr_data), daemon=True
        )
        t_out.start()
        t_err.start()

        termination: str | None = None
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )
        consecutive_pressure = 0

        try:
            while proc.poll() is None:
                if cancellation is not None and cancellation.is_set():
                    _terminate_process_tree(proc)
                    termination = "cancellation"
                    break

                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_process_tree(proc)
                    termination = "timeout"
                    break

                if memory_pressure_fn is not None and memory_pressure_fn():
                    consecutive_pressure += 1
                    if consecutive_pressure >= memory_pressure_max_backoffs:
                        _terminate_process_tree(proc)
                        termination = "oom"
                        break
                    time.sleep(memory_pressure_backoff_seconds)
                    continue

                consecutive_pressure = 0
                time.sleep(0.1)
        except BaseException:
            _terminate_process_tree(proc)
            raise
        finally:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            t_out.join(timeout=5)
            t_err.join(timeout=5)

        returncode = proc.returncode
        if returncode is None:
            returncode = -1
        return ProcessResult(
            returncode=returncode,
            stdout=stdout_data[0] if stdout_data else "",
            stderr=stderr_data[0] if stderr_data else "",
            termination=termination,
        )


@dataclass
class FakeProcessAdapter:
    _results: dict[tuple[str, ...], ProcessResult] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def enqueue(self, argv: Sequence[str], result: ProcessResult) -> None:
        self._results[tuple(argv)] = result

    def launch(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        cancellation: Event | None = None,
        memory_pressure_fn: Callable[[], bool] | None = None,
        memory_pressure_backoff_seconds: float = 1.0,
        memory_pressure_max_backoffs: int = 3,
    ) -> ProcessResult:
        key = tuple(argv)
        self.calls.append(key)
        if key not in self._results:
            raise ValueError(f"unexpected argv: {' '.join(argv)}")
        result = self._results[key]
        if cancellation is not None and cancellation.is_set():
            return ProcessResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                termination="cancellation",
            )
        if timeout_seconds is not None:
            pass
        if memory_pressure_fn is not None and memory_pressure_fn():
            return ProcessResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                termination="oom",
            )
        return result
