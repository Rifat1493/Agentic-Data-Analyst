import json

import pandas as pd
import yfinance as yf
from langchain.tools import tool


@tool("download_financial_data")
def download_financial_data(
    ticker: str,
    period: str = "1wk",
    interval: str = "1d",
) -> str:
    """Download historical OHLCV financial data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL').
        period: Lookback period. Valid values: 1d, 5d, 1wk, 1mo, 3mo, 6mo,
                1y, 2y, 5y, 10y, ytd, max. Defaults to '1wk'.
        interval: Data granularity. Valid values: 1m, 2m, 5m, 15m, 30m, 60m,
                  90m, 1h, 1d, 5d, 1wk, 1mo, 3mo. Defaults to '1d'.

    Returns:
        JSON string with the OHLCV DataFrame records, or an error message.
    """
    try:
        df: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return (
                f"No data returned for ticker '{ticker}' with "
                f"period='{period}' and interval='{interval}'."
            )

        # yfinance returns MultiIndex columns like ('Close', 'MSFT'); flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.round(4)
        df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
        return json.dumps({
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="index"),
        })
    except Exception as exc:
        return f"Error downloading data for '{ticker}': {exc}"
