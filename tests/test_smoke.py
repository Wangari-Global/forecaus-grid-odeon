def test_import():
    import forecaus_grid_odeon
    assert forecaus_grid_odeon.__version__

def test_metrics():
    from forecaus_grid_odeon.eval.metrics import mae, rmse
    assert mae([1,2,3],[1,2,3]) == 0.0
    assert rmse([0,0],[0,0]) == 0.0
