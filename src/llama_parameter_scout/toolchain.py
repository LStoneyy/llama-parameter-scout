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
# Cache type information in --help output, e.g.:
#   -ctk FNAME    ... (default: f16) {f16,f32,q8_0,q4_0}
# or a dedicated line describing supported values.
_CACHE_TYPE_HELP_RE = re.compile(
    r"-(?:ctk|ctv)\b[^\n]*?\{([^}]+)\}",
    re.MULTILINE | re.IGNORECASE,
)

# Fallback: parse option aliases from --help lines like:
#   -h, --help
#   -m, --model <FNAME>
#   --no-warmup
_HELP_OPTION_RE = re.compile(
    r"^\s{2,4}((?:-[a-zA-Z][a-zA-Z0-9]*(?:,\s*--[a-zA-Z0-9_-]+)?"
    r"|--[a-zA-Z0-9_-]+))(?=\s|<|$)",
    re.MULTILINE,
)
_HELP_OUTPUT_FORMATS_RE = re.compile(
    r"--output\b[^<]*<([a-z|]+)>",
    re.MULTILINE | re.IGNORECASE,
)


def _platform_binary_names() -> list[str]:
    if os.name == "nt":
        return ["llama-bench.exe", "llama-bench"]
    return ["llama-bench"]


def _is_executable_regular(path: Path) -> bool:
    return path.is_file() and os.access(str(path), os.X_OK)


def _sibling_cli_path(binary_path: Path) -> Path:
    parent = binary_path.parent
    if binary_path.name.endswith(".exe"):
        return parent / "llama-cli.exe"
    return parent / "llama-cli"


def _is_version_unrecognized(stderr: str, stdout: str) -> bool:
    combined = (stderr + "\n" + stdout).lower()
    mentions_version = "--version" in combined
    has_rejection = any(p in combined for p in (
        "invalid parameter",
        "unrecognized option",
        "invalid option",
        "unknown option",
        "not a valid",
        "unsupported",
    ))
    return mentions_version and has_rejection


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


def _parse_supported_flags_from_help(help_output: str) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for m in _HELP_OPTION_RE.finditer(help_output):
        raw = m.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                flags.append(part)
    if not flags:
        raise ToolchainError(
            "Could not parse any option aliases from llama-bench --help output.",
        )
    return flags


def _parse_output_formats_from_help(help_output: str) -> list[str]:
    m = _HELP_OUTPUT_FORMATS_RE.search(help_output)
    if m is None:
        return []
    return [f.strip() for f in m.group(1).split("|")]


def _parse_backends_from_help(help_output: str) -> list[str]:
    combined = help_output.lower()
    caps: list[str] = ["cpu"]
    if "ggml_metal" in combined:
        caps.append("metal")
    if "cuda" in combined:
        caps.append("cuda")
    if "hip" in combined or "rocm" in combined:
        caps.append("rocm")
    return caps


def inspect_toolchain(
    runner: ProcessRunner,
    binary_path: Path,
    help_output: str | None = None,
) -> ToolchainFingerprint:
    version_output, is_fallback = _run_version(runner, binary_path)
    if help_output is None:
        help_output = _run_help(runner, binary_path)

    if is_fallback:
        cli_name = _sibling_cli_path(binary_path).name
        version = _parse_version(version_output, label=cli_name)
        backends = _parse_backends_from_help(help_output)
        output_formats = _parse_output_formats_from_help(help_output)
        supported_flags = _parse_supported_flags_from_help(help_output)
    else:
        version = _parse_version(version_output)
        backends = _parse_backends(version_output)
        output_formats = _parse_output_formats(version_output)
        supported_flags = _parse_supported_flags(version_output)
    cache_types = _parse_cache_types(help_output, supported_flags)

    fp = ToolchainFingerprint(
        binary_path=binary_path,
        version=version,
        supported_flags=tuple(supported_flags),
        supported_output_formats=tuple(output_formats),
        device_capabilities=tuple(backends),
        supported_cache_types=tuple(cache_types),
    )

    _validate_required_flags(fp)

    return fp


def _run_version(runner: ProcessRunner, binary_path: Path) -> tuple[str, bool]:
    result = runner.run([str(binary_path), "--version"])
    if result.returncode == 0:
        if not result.stdout.strip():
            raise ToolchainError("llama-bench --version produced no output.")
        return result.stdout, False

    if not _is_version_unrecognized(result.stderr, result.stdout):
        raise ToolchainError(
            f"llama-bench --version exited with code {result.returncode}:\n"
            f"{result.stderr}",
        )

    cli_path = _sibling_cli_path(binary_path)
    cli_result = runner.run([str(cli_path), "--version"])
    if cli_result.returncode != 0:
        raise ToolchainError(
            f"llama-bench --version unrecognized; fallback {cli_path.name} --version "
            f"exited with code {cli_result.returncode}:\n"
            f"{cli_result.stderr}",
        )
    cli_output = cli_result.stdout + "\n" + cli_result.stderr
    if not cli_output.strip():
        raise ToolchainError(
            f"Fallback {cli_path.name} --version produced no output.",
        )
    return cli_output, True


def _run_help(runner: ProcessRunner, binary_path: Path) -> str:
    result = runner.run([str(binary_path), "--help"])
    if result.returncode != 0:
        raise ToolchainError(
            f"llama-bench --help exited with code {result.returncode}:\n"
            f"{result.stderr}",
        )
    combined = result.stdout + "\n" + result.stderr
    if not combined.strip():
        raise ToolchainError(
            "llama-bench --help produced no output.",
        )
    return combined


def _parse_version(output: str, label: str = "llama-bench") -> str:
    m = VERSION_RE.search(output)
    if m is None:
        raise ToolchainError(
            f"Could not find 'version:' line in {label} --version output.\n"
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


_DEFAULT_CACHE_TYPES = ("f16",)


def _parse_cache_types(help_output: str, supported_flags: list[str]) -> list[str]:
    if "-ctk" not in supported_flags and "-ctv" not in supported_flags:
        return []
    m = _CACHE_TYPE_HELP_RE.search(help_output)
    if m is not None:
        raw = m.group(1)
        types = [t.strip() for t in raw.split(",")]
        return [t for t in types if t.strip()]
    return list(_DEFAULT_CACHE_TYPES)


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
