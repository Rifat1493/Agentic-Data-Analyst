import os
import traceback
from datetime import datetime
from langchain.tools import tool

from src.middleware.guardrails import scan_generated_code


@tool("execute_python_code")
def execute_python_code(code: str) -> str:
    """Code interpreter — execute Python code and return the path to any saved chart.

    Runs the code in an isolated namespace. The code must be fully self-contained:
    include all imports, data parsing, and chart creation logic.

    Use plotly to create figures and save with: fig.write_image(OUTPUT_PATH)
    OUTPUT_PATH is pre-set in the namespace — do not hardcode a file path.

    Args:
        code: Complete, self-contained Python source code that creates exactly
              ONE plotly figure and saves it to OUTPUT_PATH.

    Returns:
        Path to the saved PNG chart, or an error message.
    """
    # GUARDRAIL: statically vet the LLM-generated code before running it.
    verdict = scan_generated_code(code)
    if not verdict.allowed:
        return (
            "Code rejected by safety guardrail (not executed):\n- "
            + "\n- ".join(verdict.violations)
        )

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(f"outputs/chart_{timestamp}.png")

    namespace: dict = {"OUTPUT_PATH": output_path}

    try:
        exec(code, namespace)  # noqa: S102
        if os.path.exists(output_path):
            return f"Chart saved to: {output_path}"
        return "Code ran without error but no chart was saved to OUTPUT_PATH."
    except Exception:
        return f"Error executing code:\n{traceback.format_exc()}"
