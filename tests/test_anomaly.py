from trainsight.analyzers.regression_anomaly import RegressionAnomalyDetector


def test_regression_anomaly_detector_flags_outlier():
    d = RegressionAnomalyDetector()
    for _ in range(30):
        d.update(50.0)
    out = d.update(95.0)
    assert out is not None
    assert out["anomaly"] is True
