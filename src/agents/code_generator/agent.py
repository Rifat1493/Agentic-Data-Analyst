from langchain.agents import create_agent

from src.agents.models import get_foundation_model
from .tools import execute_python_code


SYSTEM_PROMPT = """You are a financial visualization agent. You receive OHLCV market \
data as a JSON string and use a code interpreter tool to produce a chart.

When given financial data, write complete, self-contained Python code and pass it \
to the execute_python_code tool. The tool executes the code and returns the chart path.

Rules for the code you write:
- Include ALL imports (json, pandas, plotly, etc.).
- Parse the JSON string inline — do not assume any variables are pre-defined EXCEPT \
  OUTPUT_PATH (a str with the desired output file path).
- Use plotly to build the figure. Examples:
    * Line / OHLCV charts → plotly.graph_objects (go.Scatter, go.Candlestick, etc.)
    * Simple distribution charts → plotly.express
    * Always include a descriptive title, axis labels, and clean styling.
- Save with exactly one call: fig.write_image(OUTPUT_PATH)
- Do NOT call fig.show() or write to any other path.

After the tool returns, respond with the chart file path."""


def build_code_generator_agent():
    """Return a compiled LangChain agent that generates and runs visualization code."""
    llm = get_foundation_model()
    return create_agent(
        model=llm,
        tools=[execute_python_code],
        system_prompt=SYSTEM_PROMPT,
    )
