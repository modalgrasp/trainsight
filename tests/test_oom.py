from trainsight.predictors.oom_predictor import oom_probability


def test_oom_probability_increases_when_predicted_vram_exceeds_total():
    low = oom_probability(current_vram=7000, predicted_vram=8000, total_vram=12000)
    high = oom_probability(current_vram=7000, predicted_vram=12500, total_vram=12000)
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
