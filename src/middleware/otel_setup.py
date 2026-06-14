"""OpenTelemetry tracing setup (exports to LangSmith or any OTLP backend).

Switch it on/off entirely from .env.dev — no code changes:

    LANGSMITH_OTEL_ENABLED=true
    OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
    LANGSMITH_API_KEY=ls-...            # sent as x-api-key
    LANGSMITH_PROJECT=agentic-data-analyst
    OTEL_SERVICE_NAME=agentic-data-analyst

When disabled (the default) `configure_tracing()` is a cheap no-op and the
OpenTelemetry packages are never imported. The flag also drives LangSmith's own
OTEL emission, so enabling it makes LangGraph/LangChain spans flow to the
configured collector.

Why a tracer here at all
------------------------
A multi-agent run fans out: supervisor -> data_collector -> code_generator -> ...
Tracing stitches those nested LLM/tool/agent steps into one timeline so you can
see where latency and token spend actually go, and debug failures across agents.
"""
from __future__ import annotations

from src import config

# Module-level guard so repeated calls don't stack multiple providers.
_configured = False


def configure_tracing() -> bool:
    """Configure OTEL tracing if enabled in config. Idempotent.

    Returns:
        True if tracing is now configured, False if it was skipped (disabled or
        no endpoint).

    Raises:
        RuntimeError: if tracing is enabled but the OTEL packages are missing.
    """
    global _configured
    if _configured:
        return True
    if not config.OTEL_TRACING_ENABLED:
        return False
    if not config.OTEL_EXPORTER_OTLP_ENDPOINT:
        # Enabled but misconfigured — stay a no-op rather than crash the app.
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError(
            "OpenTelemetry tracing is enabled but dependencies are missing. "
            "Install: uv add opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        ) from exc

    # LangSmith wants the API key as x-api-key; the project header is optional.
    headers = {}
    if config.LANGSMITH_API_KEY:
        headers["x-api-key"] = config.LANGSMITH_API_KEY
    if config.LANGSMITH_PROJECT:
        headers["Langsmith-Project"] = config.LANGSMITH_PROJECT

    resource = Resource.create({"service.name": config.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{config.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces",
                headers=headers,
            )
        )
    )
    trace.set_tracer_provider(provider)

    _configured = True
    return True
