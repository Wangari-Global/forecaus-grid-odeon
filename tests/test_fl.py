import pytest
from forecaus_grid_odeon.fl import fedavg
from forecaus_grid_odeon.fl.federated import OdeonFLClient


def test_fedavg_equal_weight():
    out = fedavg([{"a": 0.0, "b": 2.0}, {"a": 2.0, "b": 4.0}])
    assert out == {"a": 1.0, "b": 3.0}


def test_fedavg_sample_weighted():
    out = fedavg([{"a": 0.0}, {"a": 3.0}], weights=[1, 2])
    assert out["a"] == pytest.approx(2.0)


def test_fedavg_key_mismatch_raises():
    with pytest.raises(ValueError):
        fedavg([{"a": 1.0}, {"b": 1.0}])


def test_odeon_client_is_stub():
    with pytest.raises(NotImplementedError):
        OdeonFLClient()
