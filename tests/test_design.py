import pandas as pd
import pytest
from patchclamp_gui.core.design import design


@pytest.mark.parametrize("animals,groups,expected", [
    (["a","a","b","c"], ["WT","WT","KO","KO"], "independent"),
    (["a","b","a","b"], ["pre","pre","post","post"], "paired"),
    (["a","b","a","c"], ["pre","pre","post","post"], "partial"),
    (["a","","b","c"], ["WT","WT","KO","KO"], "unknown"),
])
def test_design(animals, groups, expected):
    assert design(pd.DataFrame({"animal":animals,"group":groups}))[0] == expected

@pytest.mark.parametrize("animals,expected_margin", [
    (["a", "b", "c", "d"], 4.30265273 * 2**0.5),
    (["a", "b", "b", "a"], 0),
])
def test_classic_estimation_plot(animals, expected_margin):
    from types import SimpleNamespace
    from patchclamp_gui.core.design import metric_figure

    frame = pd.DataFrame({"animal": animals, "group": ["WT", "WT", "KO", "KO"],
                          "frequency": [1, 3, 5, 3]})
    figure, _ = metric_figure(SimpleNamespace(cell_table=lambda: frame), "frequency")
    groups, difference = figure.axes
    assert [bar.get_height() for bar in groups.patches] == [2, 4]
    assert difference.get_xticklabels()[0].get_text() == "KO − WT"
    interval = difference.collections[0].get_segments()[0][:, 1]
    assert interval == pytest.approx([2 - expected_margin, 2 + expected_margin])
