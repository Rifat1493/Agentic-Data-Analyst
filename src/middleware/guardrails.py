"""Guardrails — safety validation for LLM-generated content.

Use case in this system
------------------------
The code_generator agent writes Python that the code interpreter `exec()`s
in-process. A prompt-injected or hallucinating model could emit code that deletes
files, opens sockets, or escapes the sandbox. This guardrail statically inspects
that code (via Python's `ast`) BEFORE it runs and rejects anything outside a
narrow, viz-only safe set. It is a deterministic gate — never trust the LLM to
police itself.

Policy
------
* Imports: ALLOWLIST only (json, pandas, numpy, plotly, ...). Anything else
  (os, sys, subprocess, socket, requests, importlib, pickle, ...) is rejected.
* Dangerous builtins: eval, exec, compile, __import__, open, input, globals,
  locals, vars, getattr, setattr, delattr — all rejected.
* Dunder attribute access (e.g. ().__class__.__bases__) — rejected; these are the
  classic ways to break out of a restricted namespace.
"""
import ast
from dataclasses import dataclass, field
from typing import List

# Only these top-level modules may be imported by generated viz code.
SAFE_IMPORTS = {
    "json", "pandas", "numpy", "plotly", "matplotlib",
    "math", "datetime", "statistics", "collections", "itertools",
}

# Builtins that have no place in chart-drawing code and enable escapes/IO.
DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}


@dataclass
class GuardrailResult:
    allowed: bool
    violations: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.violations)


def scan_generated_code(code: str) -> GuardrailResult:
    """Statically analyse `code` and return whether it is safe to execute."""
    violations: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GuardrailResult(False, [f"Code does not parse: {exc}"])

    for node in ast.walk(tree):
        # 1. Import allowlist.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SAFE_IMPORTS:
                    violations.append(f"Disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in SAFE_IMPORTS:
                violations.append(f"Disallowed import from: {node.module}")

        # 2. Dangerous builtin calls, e.g. open(...), eval(...), __import__(...).
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_BUILTINS:
                violations.append(f"Disallowed call: {node.func.id}()")

        # 3. Dunder attribute access — sandbox-escape vector.
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(f"Disallowed dunder access: .{node.attr}")

    return GuardrailResult(len(violations) == 0, violations)
