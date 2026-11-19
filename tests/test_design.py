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
