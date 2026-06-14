"""Standalone test for the OTEL tracing setup.

Runs FULLY OFFLINE — building a TracerProvider + OTLP exporter does NOT open a
connection (export only happens on span flush, which we never trigger). We drive
the on/off behaviour by overriding `config` attributes, exactly as the .env flag
would.

Run:  python tests/test_otel.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
import src.middleware.otel_setup as otel


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


def test_disabled_is_noop():
    print("\n=== Disabled -> no-op ===")
    config.OTEL_TRACING_ENABLED = False
    otel._configured = False
    check("returns False when disabled", otel.configure_tracing() is False)


def test_enabled_without_endpoint():
    print("\n=== Enabled but no endpoint -> still a no-op ===")
    config.OTEL_TRACING_ENABLED = True
    config.OTEL_EXPORTER_OTLP_ENDPOINT = None
    otel._configured = False
    check("returns False when endpoint missing", otel.configure_tracing() is False)


def test_enabled_configures_provider():
    print("\n=== Enabled + endpoint -> configures a TracerProvider ===")
    config.OTEL_TRACING_ENABLED = True
    config.OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"
    config.OTEL_SERVICE_NAME = "test-service"
    config.LANGSMITH_PROJECT = "test-project"
    otel._configured = False

    result = otel.configure_tracing()
    check("returns True when configured", result is True)

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    check("global tracer provider is an SDK TracerProvider",
          isinstance(trace.get_tracer_provider(), TracerProvider))

    # Idempotent: a second call does not re-configure, still reports True.
    check("second call is idempotent (True)", otel.configure_tracing() is True)


if __name__ == "__main__":
    test_disabled_is_noop()
    test_enabled_without_endpoint()
    test_enabled_configures_provider()
    print("\nAll OTEL tests passed.")
