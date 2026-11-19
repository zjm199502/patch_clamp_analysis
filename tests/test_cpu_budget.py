from patchclamp_gui.core.cpu_budget import available, budget


def test_memory_cap(monkeypatch):
    import sys
    from types import SimpleNamespace
    from patchclamp_gui.core.cpu_budget import memory_lanes
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1024**3)))
    assert memory_lanes(22, 8 << 20) == 2
    assert memory_lanes(22, 1 << 30) == 1
    assert memory_lanes(1, 1024) == 1


def test_cpu_budget():
    assert available() >= 1
    assert [budget(0, count) for count in (1, 2, 8, 32)] == [1, 1, 5, 22]
    assert budget(4, 8) == 4
    assert budget(20, 8) == 8
    assert budget("invalid", 8) == 5
