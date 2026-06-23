"""Federated training across substations: fedavg determinism, the privacy
invariant (no raw samples leave a node), global-vs-local-vs-centralised, the
cold-start benefit, and round-by-round convergence."""
import numpy as np
import pandas as pd
import pytest

from forecaus_grid_odeon import config
from forecaus_grid_odeon.fl import fedavg


# ----------------------------------------------------------- fedavg reference --
def test_fedavg_is_deterministic_and_correct():
    a = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}, {"x": 5.0, "y": 6.0}]
    w = [1.0, 2.0, 3.0]
    out1 = fedavg(a, w)
    out2 = fedavg(a, w)
    assert out1 == out2                                   # no RNG -> identical
    # Matches the explicit weighted mean.
    assert out1["x"] == pytest.approx((1 * 1 + 2 * 3 + 3 * 5) / 6)
    assert out1["y"] == pytest.approx((1 * 2 + 2 * 4 + 3 * 6) / 6)


# --------------------------------------------- privacy: only parameters move --
def _synthetic_node(n_rows, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="30min", tz="UTC")
    X = {f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_features)}
    y = sum((i + 1) * X[f"f{i}"] for i in range(n_features)) + rng.normal(0, 0.1, n_rows)
    df = pd.DataFrame({"load_kw": y, **X}, index=idx)
    return df, [f"f{i}" for i in range(n_features)]


def test_no_raw_samples_leave_the_node():
    from forecaus_grid_odeon.fl import TransparentFLClient

    df, feats = _synthetic_node(n_rows=200, n_features=4)
    train, test = df.iloc[:160], df.iloc[160:]
    client = TransparentFLClient(train, test, "load_kw", feats)

    # The uplink is a fixed-length parameter vector — one float per feature —
    # NOT anything that scales with the 160 training samples.
    vecs, n_examples, metrics = client.fit([np.zeros(len(feats))], {})
    assert isinstance(vecs, list) and len(vecs) == 1
    payload = vecs[0]
    assert isinstance(payload, np.ndarray) and payload.shape == (len(feats),)
    assert payload.size == len(feats) < client.n_train     # smaller than #samples
    assert isinstance(n_examples, int) and n_examples == client.n_train  # a count, not data
    assert all(isinstance(v, float) for v in metrics.values())

    # get_parameters / evaluate also expose only parameters / scalar metrics.
    assert client.get_parameters()[0].shape == (len(feats),)
    loss, n_eval, m = client.evaluate([np.zeros(len(feats))], {})
    assert isinstance(loss, float) and isinstance(n_eval, int)
    assert set(m) == {"mape"}

    # The payload size is independent of the sample count (no per-sample leak):
    # a 10x-bigger node emits the SAME number of floats.
    big_df, _ = _synthetic_node(n_rows=1600, n_features=4, seed=1)
    big = TransparentFLClient(big_df.iloc[:1500], big_df.iloc[1500:], "load_kw", feats)
    assert big.fit([np.zeros(len(feats))], {})[0][0].size == payload.size


# -------------------------------------------------- end-to-end federation -----
@pytest.fixture(scope="module")
def fl_result(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    _orig_raw, _orig_off = config.RAW, config.OFFLINE
    config.RAW, config.OFFLINE = raw, True
    try:
        from forecaus_grid_odeon.fl import run_fl_training
        return run_fl_training(rounds=12, thin_train=24)
    finally:
        config.RAW, config.OFFLINE = _orig_raw, _orig_off


def test_more_than_one_node(fl_result):
    assert fl_result["n_nodes"] > 1                       # federation needs >1 node
    assert len(fl_result["per_feeder"]) == fl_result["n_nodes"]


def test_uplink_is_parameters_only(fl_result):
    # Every recorded uplink across all rounds/nodes was exactly n_params floats.
    assert fl_result["uplink_param_floats"] == [fl_result["n_params"]]
    assert fl_result["n_params"] < fl_result["min_node_samples"]


def test_global_beats_local_and_trails_centralised(fl_result):
    agg = fl_result["aggregate"]
    # Federation (no data shared) beats local-only and approaches the
    # data-pooled centralised upper bound: centralised <= global <= local.
    assert agg["centralised_MAPE"] <= agg["global_MAPE"] + 1e-9
    assert agg["global_MAPE"] <= agg["local_MAPE"] + 1e-9
    # Per feeder, the federated global is at least as good as local-only.
    pf = fl_result["per_feeder"]
    assert (pf["global_MAPE"] <= pf["local_MAPE"] + 1e-6).all()


def test_cold_start_benefit_is_positive(fl_result):
    cs = fl_result["cold_start"]
    assert cs["n_train"] < fl_result["per_feeder"]["n_train"].max()   # genuinely thin
    # The thin feeder gains from the shared global model.
    assert cs["benefit"] > 0
    assert cs["global_MAPE"] < cs["local_MAPE"]


def test_convergence_trace(fl_result):
    trace = fl_result["convergence"]
    assert len(trace) == 12
    assert all(np.isfinite(trace))
    # Improves over the first rounds and settles.
    assert trace[-1] < trace[0]
    assert abs(trace[-1] - trace[-2]) < 0.5               # converged (stable tail)
    # Final trace value is the aggregate global MAPE.
    assert trace[-1] == pytest.approx(fl_result["aggregate"]["global_MAPE"], abs=1e-6)


def test_federation_is_reproducible():
    """No RNG anywhere: two runs give identical metrics and trace."""
    import tempfile
    from forecaus_grid_odeon.fl import run_fl_training

    outs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            _r, _o = config.RAW, config.OFFLINE
            config.RAW, config.OFFLINE = __import__("pathlib").Path(d), True
            try:
                outs.append(run_fl_training(rounds=6, thin_train=24))
            finally:
                config.RAW, config.OFFLINE = _r, _o
    assert outs[0]["convergence"] == outs[1]["convergence"]
    pd.testing.assert_frame_equal(outs[0]["per_feeder"], outs[1]["per_feeder"])


def test_consensus_equals_fedavg_of_local_optima(fl_result):
    """The converged global == deterministic fedavg of per-node local optima
    (weighted by samples) — fedavg is the aggregation reference."""
    from forecaus_grid_odeon.fl import build_ss_nodes
    from forecaus_grid_odeon.fl.flower_client import TransparentFLClient, vector_to_params

    _r, _o = config.RAW, config.OFFLINE
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        config.RAW, config.OFFLINE = pathlib.Path(d), True
        try:
            nodes, feats, _ = build_ss_nodes(thin_train=24)
        finally:
            config.RAW, config.OFFLINE = _r, _o

    local_dicts, weights = [], []
    for tr, te in nodes.values():
        c = TransparentFLClient(tr, te, "load_kw", feats)
        local_dicts.append(vector_to_params(c.local_optimum, feats))
        weights.append(c.n_train)
    reference = fedavg(local_dicts, weights)
    for k, v in fl_result["global_params"].items():
        assert v == pytest.approx(reference[k], abs=1e-3)
