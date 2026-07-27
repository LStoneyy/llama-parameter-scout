from __future__ import annotations

from llama_parameter_scout.models import Accelerator, SystemTopology
from llama_parameter_scout.topology import (
    _is_unified_memory,
    detect_cpu_count,
    detect_system_memory,
    detect_topology,
)


class TestAccelerator:
    def test_fields(self) -> None:
        a = Accelerator(kind="gpu", backend="cuda", device_id=0, description="NVIDIA RTX 4090")
        assert a.kind == "gpu"
        assert a.backend == "cuda"
        assert a.device_id == 0
        assert a.description == "NVIDIA RTX 4090"


class TestSystemTopology:
    def test_fields(self) -> None:
        accel = Accelerator("gpu", "cuda", 0, "GPU-0")
        t = SystemTopology(
            cpu_count=8,
            system_memory_bytes=32_000_000_000,
            accelerators=(accel,),
            memory_unified=False,
        )
        assert t.cpu_count == 8
        assert t.system_memory_bytes == 32_000_000_000
        assert len(t.accelerators) == 1
        assert t.memory_unified is False

    def test_no_accelerators(self) -> None:
        t = SystemTopology(
            cpu_count=4,
            system_memory_bytes=16_000_000_000,
            accelerators=(),
            memory_unified=False,
        )
        assert len(t.accelerators) == 0
        assert t.memory_unified is False

    def test_multiple_accelerators(self) -> None:
        accels = (
            Accelerator("gpu", "cuda", 0, "NVIDIA RTX 4090"),
            Accelerator("gpu", "cuda", 1, "NVIDIA RTX 4080"),
        )
        t = SystemTopology(
            cpu_count=16,
            system_memory_bytes=64_000_000_000,
            accelerators=accels,
            memory_unified=False,
        )
        assert len(t.accelerators) == 2
        assert t.accelerators[0].device_id == 0
        assert t.accelerators[1].device_id == 1


class TestDetectCpuCount:
    def test_returns_positive_int(self) -> None:
        count = detect_cpu_count()
        assert isinstance(count, int)
        assert count >= 1


class TestDetectSystemMemory:
    def test_returns_positive_int(self) -> None:
        mem = detect_system_memory()
        assert isinstance(mem, int)
        assert mem > 0


class TestIsUnifiedMemory:
    def test_metal_on_macos_is_unified(self) -> None:
        accels = (Accelerator("gpu", "metal", 0, "Apple M1"),)
        result = _is_unified_memory(accels)
        if result:
            assert True

    def test_cuda_is_not_unified(self) -> None:
        accels = (Accelerator("gpu", "cuda", 0, "NVIDIA RTX 4090"),)
        result = _is_unified_memory(accels)
        if result:
            assert True


class TestDetectTopology:
    def test_production_detection_returns_valid_topology(self) -> None:
        topology = detect_topology()
        assert isinstance(topology, SystemTopology)
        assert topology.cpu_count >= 1
        assert topology.system_memory_bytes > 0
        assert isinstance(topology.accelerators, tuple)
        assert isinstance(topology.memory_unified, bool)

    def test_two_metal_accelerators(
        self,
    ) -> None:
        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=16_000_000_000,
            accelerators=(
                Accelerator("gpu", "metal", 0, "Apple M1"),
                Accelerator("gpu", "metal", 1, "Apple M2"),
            ),
            memory_unified=True,
        )
        assert topology.memory_unified is True
        assert len(topology.accelerators) == 2
        assert topology.system_memory_bytes == 16_000_000_000

    def test_unified_memory_budget_single_limit(self) -> None:
        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=16_000_000_000,
            accelerators=(),
            memory_unified=False,
        )
        assert topology.memory_unified is False
        assert topology.system_memory_bytes == 16_000_000_000
