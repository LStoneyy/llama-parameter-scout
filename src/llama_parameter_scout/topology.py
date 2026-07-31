from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
from collections.abc import Callable

from .models import Accelerator, SystemTopology

DetectTopologyFn = Callable[[], SystemTopology]


def detect_cpu_count() -> int:
    return os.cpu_count() or 1


def detect_system_memory() -> int:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        elif system == "Windows":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined, unused-ignore]
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.ullTotalPhys)
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        pass
    return 8 * 1024 ** 3


def detect_accelerators() -> tuple[Accelerator, ...]:
    accels: list[Accelerator] = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(", ")]
                if len(parts) >= 3:
                    idx_str, name, mem_str = parts[0], parts[1], parts[2]
                    mem_bytes: int | None = None
                    if mem_str.endswith(" MiB"):
                        try:
                            mem_bytes = int(mem_str.split()[0]) * 1024 * 1024
                        except (ValueError, IndexError):
                            pass
                    accels.append(
                        Accelerator("gpu", "cuda", int(idx_str), name, memory_bytes=mem_bytes)
                    )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    if not accels and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for gpu in data.get("SPDisplaysDataType", []):
                    name = gpu.get("sppci_model", "Unknown GPU")
                    vram_str = gpu.get("spdisplays_vram", "")
                    metal_mem: int | None = None
                    if vram_str:
                        vram_str = vram_str.replace(",", "")
                        if vram_str.endswith(" GB"):
                            try:
                                metal_mem = int(float(vram_str.split()[0]) * 1024 ** 3)
                            except (ValueError, IndexError):
                                pass
                        elif vram_str.endswith(" MB"):
                            try:
                                metal_mem = int(vram_str.split()[0]) * 1024 * 1024
                            except (ValueError, IndexError):
                                pass
                    accels.append(
                        Accelerator("gpu", "metal", len(accels), name, memory_bytes=metal_mem)
                    )
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            pass

    if not accels:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if "GPU" in line and ":" in line:
                        accels.append(
                            Accelerator("gpu", "rocm", len(accels), line)
                        )
        except (OSError, subprocess.SubprocessError):
            pass

    return tuple(accels)


def _is_unified_memory(accelerators: tuple[Accelerator, ...]) -> bool:
    if platform.system() == "Darwin":
        for a in accelerators:
            if a.backend == "metal":
                return True
    return False


def detect_topology() -> SystemTopology:
    cpu_count = detect_cpu_count()
    memory = detect_system_memory()
    accelerators = detect_accelerators()
    unified = _is_unified_memory(accelerators)
    return SystemTopology(
        cpu_count=cpu_count,
        system_memory_bytes=memory,
        accelerators=accelerators,
        memory_unified=unified,
    )
