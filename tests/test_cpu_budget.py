from patchclamp_gui.core.cpu_budget import available, budget


def test_cpu_budget():
    assert available() >= 1
    assert [budget(0, count) for count in (1, 2, 8, 32)] == [1, 1, 5, 22]
    assert budget(4, 8) == 4
    assert budget(20, 8) == 8
    assert budget("invalid", 8) == 5
