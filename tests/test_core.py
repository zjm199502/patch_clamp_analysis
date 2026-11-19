import numpy as np
import pytest

from patchclamp_gui import io
from patchclamp_gui.core import clustering, deconv, detection, stats, temporal, validation
from patchclamp_gui.core.session import CellSession, Project


def test_packaged_minmax_without_autograph():
    tf = pytest.importorskip("tensorflow")
    normalize = tf.function(detection._graph_minmax, autograph=False)
    for values, expected in (([-3., -2., 1.], [0., 0.25, 1.]), ([4., 4.], [0., 0.])):
        window = tf.constant(values)[:, None]
        np.testing.assert_allclose(normalize(window).numpy(), np.array(expected)[:, None])


def test_miniml_inference():
    pytest.importorskip("miniml")
    rec = io.simulate_recording(4, seed=1)
    result = detection.detect_miniml(rec.data, rec.fs)
    assert result.method == "miniML" and len(result.onsets) > 0
    assert len(result.onsets) == len(result.scores)
    cached = detection.detect_miniml(rec.data, rec.fs)
    np.testing.assert_array_equal(result.onsets, cached.onsets)
    np.testing.assert_allclose(result.scores, cached.scores, rtol=1e-6)
    detection.clear_model_cache()


@pytest.fixture(scope="module")
def cell():
    rec = io.simulate_recording(90, seed=3, noise_sd=1.5)
    c = CellSession(rec, group="A")
    c.params.update(detector="deconvolution")
    c.run_preprocess()
    c.run_detection()
    return c


def test_detection_recovers_events(cell):
    truth = cell.rec.meta["truth"]["onsets_s"]
    t = cell.events["time_s"].to_numpy()
    recall = np.mean([np.min(np.abs(t - x)) < 0.003 for x in truth])
    precision = np.mean([np.min(np.abs(truth - x)) < 0.003 for x in t])
    assert recall > 0.8 and precision > 0.85


def test_features_plausible(cell):
    df, w = cell.valid()
    assert len(df) == len(w)
    assert 8 < df["amplitude"].median() < 25
    assert 0.2 < df["rise_10_90"].median() < 2.5
    assert 1.5 < df["decay_tau"].median() < 12
    assert df["iei"].dropna().gt(0).all()


def test_clustering_finds_two_kinetic_populations(cell):
    df, w = cell.valid()
    keep = ~df["overlap"].to_numpy(bool)
    res = clustering.cluster_events(df[keep].reset_index(drop=True), w[keep], "waveform")
    assert res.k >= 2 and res.multi_cluster_supported
    assert res.summary["decay_tau"].iloc[-1] > 1.8 * res.summary["decay_tau"].iloc[0]


def test_clustering_single_population_not_split():
    pops = [dict(frac=1.0, amp=20, amp_cv=0.3, tau_rise=0.4, tau_decay=4.0)]
    rec = io.simulate_recording(90, seed=5, noise_sd=1.0, populations=pops)
    c = CellSession(rec); c.params.update(detector="deconvolution"); c.run_detection()
    df, w = c.valid()
    keep = ~df["overlap"].to_numpy(bool)
    res = clustering.cluster_events(df[keep].reset_index(drop=True), w[keep], "features",
                                    ["rise_10_90", "decay_tau"])
    assert res.k == 1 or not res.multi_cluster_supported


def test_deconvolution_reconstruction(cell):
    res = deconv.deconvolve(cell.processed, cell.rec.fs, 0.4, 4.0)
    base = cell.processed - np.median(cell.processed)
    assert np.std(base - res.reconvolved) < np.std(base)
    tr, td = deconv.fit_kernel_from_average(np.nanmean(cell.waves, axis=0), cell.rec.fs)
    assert 0.05 < tr < 2 and 2 < td < 12 and td > tr


def test_validation(cell):
    det = lambda y: detection.detect_deconv(y, cell.rec.fs).onsets
    curve = validation.injection_recovery(cell.processed, cell.rec.fs, det, cell.events["onset_idx"].to_numpy(),
                                          [2, 30], 0.4, 4.0, n_per_amp=20)
    assert curve["recovery"].iloc[0] < 0.5 < curve["recovery"].iloc[1]
    fp = validation.inverted_polarity_fp(cell.processed, cell.rec.fs, det, len(cell.events))
    assert fp["fp_fraction"] < 0.2


def test_temporal_poisson_vs_bursty():
    rng = np.random.default_rng(0)
    T = 300.0
    pois = np.sort(rng.uniform(0, T, 900))
    h = temporal.hawkes_fit(pois, T)
    assert h["branching_ratio"] < 0.15
    parents = np.sort(rng.uniform(0, T, 450))
    mask = rng.random(450) < 0.6
    kids = parents[mask] + rng.exponential(0.03, mask.sum())
    bursty = np.sort(np.concatenate([parents, kids]))
    h2 = temporal.hawkes_fit(bursty, T)
    assert h2["branching_ratio"] > 0.2 and h2["p_vs_poisson"] < 0.001
    ac = temporal.autocorrelogram(bursty, T, bin_ms=20, max_lag_ms=400, n_surrogate=50)
    assert ac["ratio"][0] > ac["hi"][0]


def test_scaling_multiplicative():
    rng = np.random.default_rng(1)
    c = rng.lognormal(2.8, 0.4, 2000)
    r = stats.scaling_analysis(c, rng.lognormal(2.8, 0.4, 2000) * 1.3, n=800)
    assert abs(r["iterative_factor"] - 1.3) < 0.08 and r["multiplicative"]


def test_group_stats_and_export(tmp_path):
    proj = Project()
    for i in range(6):
        pops = [dict(frac=1.0, amp=15 if i < 3 else 24, amp_cv=0.3, tau_rise=0.4, tau_decay=4.0)]
        c = CellSession(io.simulate_recording(40, seed=20 + i, noise_sd=1.5, populations=pops, name=f"c{i}"),
                        group="ctrl" if i < 3 else "trt", animal=f"m{i}")
        c.params.update(detector="deconvolution"); c.run_detection()
        proj.cells.append(c)
    res = stats.compare_groups(proj.cell_table(), "amplitude_median")
    assert res["p"] < 0.05 and res["n"] == [3, 3]
    hb = stats.hierarchical_bootstrap(proj.events_table(), "amplitude", n_boot=200)
    assert hb["p_two_sided"] < 0.05
    lm = stats.event_level_lmm(proj.events_table(), "amplitude")
    assert lm["p"] < 0.01 and 1.3 < list(lm["fold_change"].values())[0] < 2.0
    from patchclamp_gui.core import plots
    for mode in ("cell_average", "pooled_equal", "pooled_all"):
        fig, txt = plots.cumulative_figure(proj, mode=mode)
        assert len(fig.axes) == 2 and "p =" in txt
    fig.savefig(tmp_path / "cdf.pdf")
    files = proj.export(str(tmp_path / "out"))
    assert all((tmp_path / f.split("/")[-1]).exists() for f in files)


