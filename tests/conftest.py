"""Shared pytest setup.

Makes the project importable as `import src...` / `import app` when pytest is
run from anywhere, and supplies a dummy LLM key so modules that *construct* (but
never call) a chat model at import time don't blow up in offline/CI runs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Construction-only safety net — no network call is ever made with this.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-offline")
