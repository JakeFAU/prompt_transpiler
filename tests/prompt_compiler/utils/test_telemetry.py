from prompt_compiler.config import settings
from prompt_compiler.utils.telemetry import TelemetryManager


def test_telemetry_disabled(monkeypatch, mocker):
    """Test that telemetry setup is skipped when disabled."""
    # Disable telemetry
    monkeypatch.setenv("PRCOMP_USE_OPENTELEMETRY", "false")
    settings.reload()

    # Mock dependencies to verify they are NOT called
    mock_resource = mocker.patch("prompt_compiler.utils.telemetry.Resource")
    mock_provider = mocker.patch("prompt_compiler.utils.telemetry.TracerProvider")

    tm = TelemetryManager()
    tm.setup()

    # Should not be called
    mock_resource.create.assert_not_called()
    mock_provider.assert_not_called()


def test_telemetry_enabled(monkeypatch, mocker):
    """Test that telemetry setup proceeds when enabled."""
    # Enable telemetry
    monkeypatch.setenv("PRCOMP_USE_OPENTELEMETRY", "true")
    # Ensure we don't actually try to connect to anything
    monkeypatch.setenv("PRCOMP_OPENTEL__OTEL_ENDPOINT", "")
    settings.reload()

    # Mock dependencies
    mock_resource = mocker.patch("prompt_compiler.utils.telemetry.Resource")
    mock_provider = mocker.patch("prompt_compiler.utils.telemetry.TracerProvider")
    # Mock the ConsoleSpanExporter since we cleared the endpoint
    mocker.patch("prompt_compiler.utils.telemetry.ConsoleSpanExporter")
    mock_trace = mocker.patch("prompt_compiler.utils.telemetry.trace")
    mock_metrics = mocker.patch("prompt_compiler.utils.telemetry.metrics")

    tm = TelemetryManager()
    tm.setup()

    # Should be called
    mock_resource.create.assert_called_once()
    mock_provider.assert_called_once()
    mock_trace.set_tracer_provider.assert_called_once()
    mock_trace.get_tracer.assert_called()
    mock_metrics.get_meter.assert_called()


def test_telemetry_span_decorator(monkeypatch, mocker):
    """Test the instrument decorator."""
    monkeypatch.setenv("PRCOMP_USE_OPENTELEMETRY", "true")
    settings.reload()

    tm = TelemetryManager()
    # We need to mock the tracer inside the manager or the global trace
    mock_tracer = mocker.MagicMock()
    mock_span = mocker.MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

    mocker.patch("prompt_compiler.utils.telemetry.trace.get_tracer", return_value=mock_tracer)

    @tm.instrument(name="test_func")
    def my_func(x):
        return x * 2

    result = my_func(5)

    assert result == 10  # noqa: PLR2004
    mock_tracer.start_as_current_span.assert_called_with("test_func", attributes={})
