"""
HTML report for the strategy ranking (backtesting-rank).
"""

import logging
from datetime import datetime
from math import ceil, floor, log10
from pathlib import Path
from typing import Any

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.util import dt_now, render_template


logger = logging.getLogger(__name__)


def nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    """
    Round axis ticks covering low..high (always including 0 when in range).
    """
    if high <= low:
        high = low + 1.0
    raw_step = (high - low) / count
    magnitude = 10 ** floor(log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    start = floor(low / step) * step
    stop = ceil(high / step) * step
    ticks = []
    value = start
    while value <= stop + step / 2:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt_axis(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def cumulative_profit_series(
    trades: list[dict[str, Any]], start: datetime, end: datetime
) -> list[tuple[int, float]]:
    """
    Cumulative absolute profit over time, as (timestamp in ms, profit) points.
    Starts at 0 at the backtest start and keeps the final value until the backtest end.
    """
    points = [(int(start.timestamp() * 1000), 0.0)]
    total = 0.0
    for trade in sorted(trades, key=lambda t: t["close_timestamp"]):
        total += trade["profit_abs"]
        points.append((int(trade["close_timestamp"]), total))
    end_ts = int(end.timestamp() * 1000)
    if points[-1][0] < end_ts:
        points.append((end_ts, total))
    return points


def build_line_chart(
    series: list[dict[str, Any]], start: datetime, end: datetime
) -> dict[str, Any]:
    """
    Chart geometry for the template - x / y are percentages of the plot area.
    :param series: list of {"name": str, "slot": int, "points": [(ts_ms, value)]}
    """
    start_ts = int(start.timestamp() * 1000)
    end_ts = max(int(end.timestamp() * 1000), start_ts + 1)
    values = [v for s in series for _, v in s["points"]]
    ticks = nice_ticks(min(0.0, *values), max(0.0, *values))
    y_low, y_high = ticks[0], ticks[-1]

    def _x(ts: int) -> float:
        return round(min(max((ts - start_ts) / (end_ts - start_ts), 0.0), 1.0) * 100, 3)

    def _y(value: float) -> float:
        return round((y_high - value) / (y_high - y_low) * 100, 3)

    chart_series = []
    for s in series:
        points = [{"x": _x(ts), "y": _y(v), "v": round(v, 4), "t": ts} for ts, v in s["points"]]
        chart_series.append(
            {
                "name": s["name"],
                "slot": s["slot"],
                "polyline": " ".join(f"{p['x']},{p['y']}" for p in points),
                "points": points,
                "last": points[-1],
            }
        )
    return {
        "series": chart_series,
        "ticks": [{"y": _y(t), "label": _fmt_axis(t), "zero": t == 0} for t in ticks],
        "start": start.strftime("%Y-%m-%d %H:%M"),
        "end": end.strftime("%Y-%m-%d %H:%M"),
    }


def build_bar_rows(rows: list[dict[str, Any]]) -> None:
    """
    Add diverging bar geometry (percent of the track) for profit_total to each ranking row.
    """
    profits = [r["profit_total"] for r in rows]
    low, high = min(0.0, *profits), max(0.0, *profits)
    span = (high - low) or 1.0
    zero = (0 - low) / span * 100
    for row in rows:
        value = row["profit_total"]
        width = abs(value) / span * 100
        row["bar_zero"] = round(zero, 3)
        row["bar_left"] = round(zero - width if value < 0 else zero, 3)
        row["bar_width"] = round(width, 3)


def render_ranking_report(data: dict[str, Any]) -> str:
    build_bar_rows(data["ranking"])
    return render_template("strategy_ranking_report.html.j2", data)


def store_ranking_report(config: dict[str, Any], data: dict[str, Any]) -> Path:
    """
    Write the HTML report to the backtest results directory.
    :return: Path of the written report
    """
    directory = Path(config["exportdirectory"])
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"backtest-rank-{dt_now():%Y-%m-%d_%H-%M-%S}.html"
    data.setdefault("generated", dt_now().strftime(DATETIME_PRINT_FORMAT))
    filename.write_text(render_ranking_report(data), encoding="utf-8")
    return filename
