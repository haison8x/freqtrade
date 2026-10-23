from datetime import UTC, datetime

from freqtrade.optimize.strategy_ranking_report import (
    build_line_chart,
    cumulative_profit_series,
    nice_ticks,
    render_ranking_report,
    store_ranking_report,
)


START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)


def _ts(hour: int) -> int:
    return int(datetime(2024, 1, 1, hour, tzinfo=UTC).timestamp() * 1000)


def test_nice_ticks():
    assert nice_ticks(-3.2, 7.9) == [-5, 0, 5, 10]
    assert nice_ticks(0.0, 0.05) == [0.0, 0.02, 0.04, 0.06]
    assert nice_ticks(0.0, 1000.0) == [0.0, 250.0, 500.0, 750.0, 1000.0]
    # Flat series
    assert nice_ticks(0.0, 0.0) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_cumulative_profit_series():
    trades = [
        {"close_timestamp": _ts(10), "profit_abs": -2.0},
        {"close_timestamp": _ts(5), "profit_abs": 5.0},
    ]
    assert cumulative_profit_series(trades, START, END) == [
        (_ts(0), 0.0),
        (_ts(5), 5.0),
        (_ts(10), 3.0),
        (int(END.timestamp() * 1000), 3.0),
    ]
    assert cumulative_profit_series([], START, END) == [
        (_ts(0), 0.0),
        (int(END.timestamp() * 1000), 0.0),
    ]


def test_build_line_chart():
    points = [(_ts(0), 0.0), (_ts(12), -10.0), (int(END.timestamp() * 1000), 30.0)]
    chart = build_line_chart([{"name": "Current", "slot": 1, "points": points}], START, END)

    assert [t["label"] for t in chart["ticks"]] == ["-10", "0", "10", "20", "30"]
    assert [t["zero"] for t in chart["ticks"]] == [False, True, False, False, False]
    line = chart["series"][0]
    assert line["polyline"] == "0.0,75.0 50.0,100.0 100.0,0.0"
    assert line["last"] == {"x": 100.0, "y": 0.0, "v": 30.0, "t": points[-1][0]}
    assert chart["start"] == "2024-01-01 00:00"
    assert chart["end"] == "2024-01-02 00:00"


def _report_data(name: str = "MyStrategy") -> dict:
    points = [(_ts(0), 0.0), (_ts(5), 12.5)]
    return {
        "hours": 24,
        "window_start": "2024-01-01 00:00:00",
        "window_end": "2024-01-01 23:55:00",
        "rank_by": "profit",
        "stake_currency": "USDT",
        "pairs": ["BTC/USDT", "ETH/USDT"],
        "tiles": [{"label": "Best strategy", "value": name, "note": "1.25%", "sign": 0}],
        "metric_header": None,
        "ranking": [
            {
                "rank": 1,
                "name": name,
                "timeframe": "5m",
                "trades": 3,
                "wdl": "2 / 0 / 1",
                "winrate": "66.7%",
                "profit_total": 0.0125,
                "profit_pct": "1.25%",
                "profit_abs": "12.5 USDT",
                "sign": 1,
                "result": "PROFIT",
                "max_dd": "0.50%",
                "metric": "0.01",
            },
            {
                "rank": 2,
                "name": "Loser",
                "timeframe": "1h",
                "trades": 2,
                "wdl": "0 / 0 / 2",
                "winrate": "0.0%",
                "profit_total": -0.005,
                "profit_pct": "-0.50%",
                "profit_abs": "-5 USDT",
                "sign": -1,
                "result": "LOSS",
                "max_dd": "0.50%",
                "metric": "-0.01",
            },
        ],
        "strategies": [
            {
                "rank": 1,
                "name": name,
                "timeframe": "5m",
                "period": "2024-01-01 00:00:00 -> 2024-01-01 23:55:00",
                "sign": 1,
                "result": "PROFIT",
                "details": [{"label": "Total profit", "value": "12.5 USDT (1.25%)", "sign": 1}],
                "chart": build_line_chart(
                    [{"name": "Current parameters", "slot": 1, "points": points}], START, END
                ),
                "suggestions": [
                    {
                        "name": "stoploss",
                        "current": "-0.1",
                        "recommended": "-0.05",
                        "applied": True,
                        "reason": "test reason",
                    }
                ],
                "tuned": "Backtest with recommended parameters: profit 2.00%",
                "params_code": "    stoploss = -0.05",
                "hyperopt_params": "buy_rsi=30",
                "hyperopt_cmd": "freqtrade hyperopt --strategy MyStrategy",
                "notes": ["A note"],
            }
        ],
    }


def test_render_ranking_report():
    html = render_ranking_report(_report_data())
    assert html.startswith("<!doctype html>")
    assert "<title>Strategy ranking</title>" in html
    assert "Last 24 hours" in html
    assert "#1 MyStrategy" in html
    assert "stoploss = -0.05" in html
    assert "freqtrade hyperopt --strategy MyStrategy" in html
    assert "test reason" in html
    # Diverging bars: profit right of zero, loss left of it
    assert "left: 28.571%; width: max(71.429%, 2px)" in html
    assert "left: 0.0%; width: max(28.571%, 2px)" in html
    # Single series - no legend
    assert '<div class="legend">' not in html


def test_render_ranking_report_escapes():
    html = render_ranking_report(_report_data("<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_store_ranking_report(tmp_path, time_machine):
    time_machine.move_to(datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC), tick=False)
    path = store_ranking_report({"exportdirectory": tmp_path / "results"}, _report_data())
    assert path == tmp_path / "results" / "backtest-rank-2024-01-02_03-04-05.html"
    assert "Generated by freqtrade backtesting-rank on 2024-01-02 03:04:05" in path.read_text(
        encoding="utf-8"
    )
