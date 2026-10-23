
from src.datapilot.health import get_health


def test_health_returns_expected_values():
    result = get_health()

    assert result["status"] == "ok"
    assert result["app"] == "DataPilot AI"
    assert result["environment"] == "development"
