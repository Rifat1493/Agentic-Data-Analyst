"""Central configuration — the ONE place we load .env and read environment.

Every other module imports this and reads `config.<NAME>` at call time, e.g.:

    from src import config
    url = config.POSTGRES_URL

Reading `config.NAME` (rather than copying the value at import) means tests can
override a setting by assigning to the attribute, and the change is picked up.
"""
import os

from dotenv import load_dotenv

# Load .env.dev once for the whole process. All env reads happen below.
load_dotenv(".env.dev")


# --------------------------------------------------------------------------- #
# LLM / DashScope (Qwen)                                                       #
# --------------------------------------------------------------------------- #
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
FOUNDATION_MODEL = os.getenv("FOUNDATION_MODEL", "qwen3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


# --------------------------------------------------------------------------- #
# Databases / memory backends                                                  #
# --------------------------------------------------------------------------- #
POSTGRES_URL = os.getenv("POSTGRES_URL")
REDIS_URL = os.getenv("REDIS_URL")


# --------------------------------------------------------------------------- #
# Auth0 (authentication)                                                       #
# --------------------------------------------------------------------------- #
DEFAULT_ROLES_CLAIM = "https://agentic-data-analyst/roles"

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
AUTH0_ISSUER = os.getenv("AUTH0_ISSUER") or (
    f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None
)
AUTH0_ROLES_CLAIM = os.getenv("AUTH0_ROLES_CLAIM", DEFAULT_ROLES_CLAIM)


# --------------------------------------------------------------------------- #
# Rate limiting                                                                 #
# --------------------------------------------------------------------------- #
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


# --------------------------------------------------------------------------- #
# Semantic cache                                                                #
# --------------------------------------------------------------------------- #
SEMANTIC_CACHE_DISTANCE = float(os.getenv("SEMANTIC_CACHE_DISTANCE", "0.1"))
SEMANTIC_CACHE_TTL = int(os.getenv("SEMANTIC_CACHE_TTL")) if os.getenv("SEMANTIC_CACHE_TTL") else None


# --------------------------------------------------------------------------- #
# Observability — OpenTelemetry tracing (LangSmith)                            #
# --------------------------------------------------------------------------- #
# Flip OTEL_TRACING on/off purely from .env.dev. LangSmith itself also reads
# LANGSMITH_OTEL_ENABLED from the environment, so the same flag drives both.
OTEL_TRACING_ENABLED = os.getenv("LANGSMITH_OTEL_ENABLED", "false").lower() == "true"
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")  # base URL, no /v1/traces
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "agentic-data-analyst")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
