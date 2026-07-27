from __future__ import annotations

import os
import re
from pathlib import Path

from .models import BinaryResolution, ToolchainError, ToolchainFingerprint
from .runner import ProcessRunner

VERSION_RE = re.compile(r"^version:\s*(\S+)", re.MULTILINE)
BACKENDS_RE = re.compile(r"^backends:[ \t]*(.*)$", re.MULTILINE)
OUTPUT_FORMATS_RE = re.compile(r"^supported_output_formats:[ \t]*(.*)$", re.MULTILINE)
SUPPORTED_FLAGS_RE = re.compile(r"^supported_flags:[ \t]*(.*)$", re.MULTILINE)


def _platform_binary_names() -> list[str]:
    if os.name == "nt":
        return ["llama-bench.exe", "llama-bench"]
    return ["llama-bench"]


def _is_executable_regular(path: Path) -> bool:
    return path.is_file() and os.access(str(path), os.X_OK)


def resolve_binary(
    llama_bench: Path | None = None,
    install_dir: Path | None = None,
) -> BinaryResolution:
    if llama_bench is not None:
        return BinaryResolution(binary_path=llama_bench, source="explicit")
    if install_dir is not None:
        for name in _platform_binary_names():
            candidate = install_dir / name
            if _is_executable_regular(candidate):
                return BinaryResolution(binary_path=candidate, source="install_dir")
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        for name in _platform_binary_names():
            candidate = Path(p) / name
            if _is_executable_regular(candidate):
                return BinaryResolution(binary_path=candidate, source="path")
    raise ToolchainError(
        "llama-bench not found. Provide --llama-bench, --install-dir, "
        "or ensure it is on PATH.",
    )


def inspect_toolchain(
    runner: ProcessRunner,
    binary_path: Path,
) -> ToolchainFingerprint:
    version_output = _run_version(runner, binary_path)
    _run_help(runner, binary_path)

    version = _parse_version(version_output)
    backends = _parse_backends(version_output)
    output_formats = _parse_output_formats(version_output)
    supported_flags = _parse_supported_flags(version_output)

    fp = ToolchainFingerprint(
        binary_path=binary_path,
        version=version,
        supported_flags=tuple(supported_flags),
        supported_output_formats=tuple(output_formats),
        device_capabilities=tuple(backends),
    )

    _validate_required_flags(fp)

    return fp


def _run_version(runner: ProcessRunner, binary_path: Path) -> str:
    result = runner.run([str(binary_path), "--version"])
    if result.returncode != 0:
        raise ToolchainError(
            f"llama-bench --version exited with code {result.returncode}:\n"
            f"{result.stderr}",
        )
    if not result.stdout.strip():
        raise ToolchainError("llama-bench --version produced no output.")
    return result.stdout


def _run_help(runner: ProcessRunner, binary_path: Path) -> None:
    result = runner.run([str(binary_path), "--help"])
    if result.returncode != 0:
        raise ToolchainError(
            f"llama-bench --help exited with code {result.returncode}:\n"
            f"{result.stderr}",
        )
    if not result.stdout.strip():
        raise ToolchainError(
            "llama-bench --help produced no output.",
        )


def _parse_version(output: str) -> str:
    m = VERSION_RE.search(output)
    if m is None:
        raise ToolchainError(
            "Could not find 'version:' line in llama-bench --version output.\n"
            f"Output:\n{output}",
        )
    return m.group(1)


def _parse_backends(output: str) -> list[str]:
    m = BACKENDS_RE.search(output)
    if m is None:
        raise ToolchainError(
            "Missing 'backends:' line in llama-bench --version output.\n"
            f"Output:\n{output}",
        )
    raw = m.group(1).strip()
    if not raw:
        raise ToolchainError(
            "Empty 'backends:' line in llama-bench --version output.",
        )
    return [b.strip() for b in raw.split(",")]


def _parse_output_formats(output: str) -> list[str]:
    m = OUTPUT_FORMATS_RE.search(output)
    if m is None:
        raise ToolchainError(
            "Missing 'supported_output_formats:' line in llama-bench --version output.\n"
            f"Output:\n{output}",
        )
    raw = m.group(1).strip()
    if not raw:
        raise ToolchainError(
            "Empty 'supported_output_formats:' line in llama-bench --version output.",
        )
    return [f.strip() for f in raw.split(",")]


def _parse_supported_flags(output: str) -> list[str]:
    m = SUPPORTED_FLAGS_RE.search(output)
    if m is None:
        raise ToolchainError(
            "Missing 'supported_flags:' line in llama-bench --version output.\n"
            f"Output:\n{output}",
        )
    raw = m.group(1).strip()
    if not raw:
        raise ToolchainError(
            "Empty 'supported_flags:' line in llama-bench --version output.",
        )
    return [f.strip() for f in raw.split(",")]


_REQUIRED_FLAGS = frozenset({
    "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
    "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
})


def _validate_required_flags(fp: ToolchainFingerprint) -> None:
    present = set(fp.supported_flags)
    missing = _REQUIRED_FLAGS - present
    if missing:
        raise ToolchainError(
            f"llama-bench at {fp.binary_path} is missing required supported flags: "
            f"{', '.join(sorted(missing))}",
        )
    if "json" not in fp.supported_output_formats:
        raise ToolchainError(
            f"llama-bench at {fp.binary_path} does not support 'json' output format. "
            f"Supported: {', '.join(fp.supported_output_formats)}",
        )
