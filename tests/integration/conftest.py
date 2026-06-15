import pytest
import app.services.forecast as forecast_svc
import app.services.surge as surge_svc
import app.services.incident as incident_svc


@pytest.fixture(autouse=True)
def reset_forecast_state():
    """Clear all stateful service windows before each integration test."""
    forecast_svc._windows.clear()
    forecast_svc._active_drivers.clear()
    forecast_svc._last_emit.clear()
    surge_svc._history.clear()
    surge_svc._last_surge.clear()
    incident_svc._anomaly_window.clear()
    incident_svc._last_incident.clear()
    yield
