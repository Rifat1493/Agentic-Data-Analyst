"""Standalone tests for the code-safety guardrail.

Fully OFFLINE — pure static analysis, no LLM/network. Verifies that legitimate
viz code passes and that dangerous code is rejected both by the scanner and by
the code-interpreter tool (which must NOT exec rejected code).

Run:  python tests/test_guardrails.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.middleware.guardrails import scan_generated_code


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


# A realistic, safe plotly snippet like the code_generator produces.
SAFE_CODE = """
import json
import pandas as pd
import plotly.graph_objects as go

data = json.loads('{"data": {"2026-06-09": {"Close": 450.2}}}')
df = pd.DataFrame(data["data"]).T
fig = go.Figure(go.Scatter(x=df.index, y=df["Close"]))
fig.update_layout(title="MSFT")
fig.write_image(OUTPUT_PATH)
"""


def test_safe_code_allowed():
    print("\n=== Safe viz code is allowed ===")
    r = scan_generated_code(SAFE_CODE)
    check("safe plotly code passes", r.allowed)
    check("no violations reported", r.violations == [])


def test_dangerous_code_blocked():
    print("\n=== Dangerous code is blocked ===")
    cases = {
        "os import": "import os\nos.system('echo hacked')",
        "subprocess import": "import subprocess\nsubprocess.run(['ls'])",
        "from-import socket": "from socket import socket",
        "open() file write": "open('/etc/passwd', 'w').write('x')",
        "eval()": "eval('2 + 2')",
        "__import__ bypass": "__import__('os').system('id')",
        "dunder escape": "x = ().__class__.__bases__[0].__subclasses__()",
        "getattr bypass": "getattr(__builtins__, 'eval')('1')",
        "syntax error": "import (((",
    }
    for label, code in cases.items():
        r = scan_generated_code(code)
        check(f"blocked: {label}", not r.allowed and len(r.violations) >= 1)


def test_tool_does_not_execute_rejected_code():
    print("\n=== The code interpreter refuses to exec rejected code ===")
    from src.agents.code_generator.tools import execute_python_code

    # A clearly malicious payload; must be rejected, never executed.
    malicious = "import os\nos.system('echo SHOULD_NOT_RUN')"
    result = execute_python_code.invoke({"code": malicious})
    check("tool returns a rejection message", "rejected by safety guardrail" in result)
    check("tool did NOT report a saved chart", "Chart saved to" not in result)


if __name__ == "__main__":
    test_safe_code_allowed()
    test_dangerous_code_blocked()
    test_tool_does_not_execute_rejected_code()
    print("\nAll guardrail tests passed.")
