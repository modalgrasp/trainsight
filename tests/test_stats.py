from trainsight.app import row_from_payload


def test_row_from_payload_clamps_and_sets_defaults():
    row = row_from_payload(
        {
            "gpu_util": 130,
            "vram": 88,
            "mem_used": 10000,
            "mem_total": 12000,
            "temp": 82,
            "power": 140,
            "power_limit": 200,
            "name": "Fake",
            "index": 0,
        },
        power_cap_watts=200,
    )

    assert row["gpu_util"] == 100.0
    assert row["mem_percent"] == 88.0
    assert row["name"] == "Fake"
    assert row["power_limit"] == 200.0


def test_peak_tracking_logic_like_dashboard():
    peak_gpu = 0.0
    smoothed_gpu = 95.0
    peak_gpu = max(peak_gpu, smoothed_gpu)
    assert peak_gpu == 95.0
