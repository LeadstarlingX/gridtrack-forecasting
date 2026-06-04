import pytest
import app.services.forecast as forecast_svc


@pytest.fixture(autouse=True)
def reset_forecast_state():
    """Clear sliding-window state before each integration test so the
    throttle and event windows are always clean."""
    forecast_svc._windows.clear()
    forecast_svc._active_drivers.clear()
    forecast_svc._last_emit.clear()
    yield
