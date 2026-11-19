import shutil

import numpy as np
import pandas as pd
import pytest

from patchclamp_gui import io
from patchclamp_gui.core import project_files
from patchclamp_gui.core.session import DEFAULT_PARAMS, CellSession


def test_portable_project(tmp_path):
    root = tmp_path / "实验"
    project = project_files.create_project(root)
    with pytest.raises(FileExistsError):
        project_files.create_project(root)
    source = tmp_path / "数据.npy"
    np.save(source, io.simulate_recording(5, seed=18).data)
    raw = project_files.copy_raw(root, source)
    assert project_files.copy_raw(root, source) != raw
    cell = CellSession(io.load_any(str(raw)), group="对照", animal="A1")
    cell.params.update(detector="deconvolution")
    assert (cell.params["lowpass_hz"], cell.params["notch_hz"]) == (1000, 50)
    assert cell.run_detection() > 1
    cell.events.loc[0, ["excluded", "cluster"]] = [True, 2]
    project.cells.append(cell)
    project_files.save_project(root, project)
    moved = tmp_path / "移位"
    shutil.move(str(root), moved); source.unlink()
    reopened = project_files.open_project(moved).cells[0]
    assert reopened.group == cell.group and reopened.animal == cell.animal and reopened.params == cell.params
    np.testing.assert_array_equal(reopened.rec.data, cell.rec.data)
    np.testing.assert_array_equal(reopened.processed, cell.processed)
    np.testing.assert_array_equal(reopened.waves, cell.waves)
    np.testing.assert_array_equal(reopened.det.onsets, cell.det.onsets)
    pd.testing.assert_frame_equal(reopened.events, cell.events)


def test_failed_save_preserves_project(tmp_path):
    root = tmp_path / "project"
    project = project_files.create_project(root)
    cell = CellSession(io.simulate_recording(1), params=dict(DEFAULT_PARAMS))
    project.cells.append(cell); project_files.save_project(root, project)
    cell.rec.path = str(tmp_path / "outside.npy")
    with pytest.raises(ValueError):
        project_files.save_project(root, project)
    restored = project_files.open_project(root).cells[0]
    assert restored.params == cell.params and restored.events is None
