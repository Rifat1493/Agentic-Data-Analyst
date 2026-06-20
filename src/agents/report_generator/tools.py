import os
from datetime import datetime

from langchain.tools import tool


@tool("save_report")
def save_report(markdown: str) -> str:
    """Save a finished financial analysis report to disk as a Markdown file.

    Args:
        markdown: The complete report content formatted as Markdown.

    Returns:
        Path to the saved Markdown file.
    """
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(f"outputs/report_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return f"Report saved to: {output_path}"
