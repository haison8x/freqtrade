from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from freqtrade.commands.optimize_commands import (
    setup_optimize_configuration,
    start_backtesting_rank,
)
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.strategy_ranking import (
    ParamSuggestion,
    StrategyRanking,
    calculate_trade_excursions,
    get_lookback_window,
    rank_strategies,
    recommend_roi,
    recommend_stoploss,
    recommend_trailing,
)
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    get_args,
    patch_exchange,
    patched_configuration_load_config_file,
)


def _excursions(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["mae", "mfe", "profit_ratio", "trade_duration"])


def test_get_lookback_window():
    now = datetime(2024, 5, 10, 12, 17, 33, tzinfo=UTC)
    start, end = get_lookback_window(24, "5m", now)
    # 12:15 candle is still open
    assert end == datetime(2024, 5, 10, 12, 10, tzinfo=UTC)
    assert start == datetime(2024, 5, 9, 12, 15, tzinfo=UTC)

    start, end = get_lookback_window(6, "1h", now)
    assert end == datetime(2024, 5, 10, 11, 0, tzinfo=UTC)
    assert start == datetime(2024, 5, 10, 6, 0, tzinfo=UTC)


def test_calculate_trade_excursions():
    trades = [
        # Long - dipped 2%, ran up 5%
        {
            "open_rate": 100.0,
            "min_rate": 98.0,
            "max_rate": 105.0,
            "is_short": False,
            "leverage": 1.0,
            "profit_ratio": 0.03,
            "trade_duration": 60,
        },
        # Short with leverage 2 - price went up 1% (adverse), down 4% (favorable)
        {
            "open_rate": 100.0,
            "min_rate": 96.0,
            "max_rate": 101.0,
            "is_short": True,
            "leverage": 2.0,
            "profit_ratio": 0.05,
            "trade_duration": 30,
        },
        # Missing min / max rate
        {
            "open_rate": 100.0,
            "min_rate": None,
            "max_rate": None,
            "is_short": False,
            "leverage": 1.0,
            "profit_ratio": 0.0,
            "trade_duration": 5,
        },
    ]
    res = calculate_trade_excursions(trades)
    assert res["mae"].tolist() == pytest.approx([-0.02, -0.02, 0.0])
    assert res["mfe"].tolist() == pytest.approx([0.05, 0.08, 0.0])
    assert res["profit_ratio"].tolist() == pytest.approx([0.03, 0.05, 0.0])
    assert res["leverage"].tolist() == [1.0, 2.0, 1.0]

    assert calculate_trade_excursions([]).empty


def test_recommend_stoploss():
    excursions = _excursions(
        [
            (-0.01, 0.03, 0.02, 30),
            (-0.012, 0.04, 0.03, 30),
            (-0.008, 0.02, 0.01, 30),
            (-0.10, 0.0, -0.10, 60),
            (-0.10, 0.005, -0.10, 60),
        ]
    )
    res = recommend_stoploss(excursions, -0.10)
    assert res is not None
    assert res.name == "stoploss"
    assert res.current == -0.10
    assert -0.03 < res.recommended < -0.01
    assert "2 losing trades" in res.reason

    # Leverage 5 - deep dips are capped well before liquidation
    leveraged = excursions.assign(mae=excursions["mae"] * 20, leverage=5.0)
    res_lev = recommend_stoploss(leveraged, -0.5)
    assert res_lev is not None
    # 90% quantile 0.232 * 1.2 + 0.005 * 5
    assert res_lev.recommended == -0.303
    res_lev = recommend_stoploss(leveraged.assign(mae=excursions["mae"] * 60), -0.5)
    assert res_lev is not None
    assert res_lev.recommended == -0.8

    # Already close to the recommendation
    assert recommend_stoploss(excursions, res.recommended) is None
    # Not enough winners
    assert recommend_stoploss(excursions.iloc[2:], -0.10) is None


def test_recommend_roi():
    excursions = _excursions(
        [
            (-0.01, 0.03, 0.02, 30),
            (-0.01, 0.04, 0.01, 60),
            (-0.01, 0.02, 0.01, 40),
            (-0.02, 0.01, -0.01, 50),
            (-0.03, 0.025, -0.02, 20),
        ]
    )
    res = recommend_roi(excursions, {0: 0.10}, "5m", 0.001)
    assert res is not None
    assert res.current == {"0": 0.10}
    assert res.recommended == {"0": 0.025, "40": 0.0125, "80": 0.0063, "160": 0}

    # Same as current
    assert recommend_roi(excursions, res.recommended, "5m", 0.001) is None
    # Moves too small compared to fees
    assert recommend_roi(excursions, {0: 0.10}, "5m", 0.01) is None
    # Not enough trades
    assert recommend_roi(excursions.iloc[:3], {0: 0.10}, "5m", 0.001) is None


def test_recommend_trailing():
    current = {
        "trailing_stop": False,
        "trailing_stop_positive": None,
        "trailing_stop_positive_offset": 0.0,
        "trailing_only_offset_is_reached": False,
    }
    # Winners gave back most of their move
    excursions = _excursions(
        [
            (-0.01, 0.06, 0.01, 30),
            (-0.01, 0.08, 0.02, 30),
            (-0.01, 0.05, 0.01, 30),
            (-0.02, 0.01, -0.02, 30),
        ]
    )
    res = recommend_trailing(excursions, current, 0.001)
    assert res is not None
    assert res.recommended == {
        "trailing_stop": True,
        "trailing_stop_positive": 0.015,
        "trailing_stop_positive_offset": 0.03,
        "trailing_only_offset_is_reached": True,
    }
    assert res.as_strategy_params() == res.recommended

    # Winners kept most of their move
    excursions["profit_ratio"] = [0.05, 0.07, 0.04, -0.02]
    assert recommend_trailing(excursions, current, 0.001) is None


def test_rank_strategies():
    results = {
        "Loser": {"profit_total": -0.05, "total_trades": 10, "wins": 2, "losses": 8},
        "Winner": {"profit_total": 0.10, "total_trades": 10, "wins": 8, "losses": 2},
        "FewTrades": {"profit_total": 0.50, "total_trades": 1, "wins": 1, "losses": 0},
        "NoTrades": {"profit_total": 0.0, "total_trades": 0, "wins": 0, "losses": 0},
    }
    ranked = [name for name, _ in rank_strategies(results, "profit", 1)]
    assert ranked == ["FewTrades", "Winner", "Loser", "NoTrades"]

    ranked = [name for name, _ in rank_strategies(results, "profit", 5)]
    assert ranked == ["Winner", "Loser", "FewTrades", "NoTrades"]

    results["Winner"]["profit_factor"] = 3.0
    results["Loser"]["profit_factor"] = 0.4
    # No losses - profit factor is reported as 0
    results["FewTrades"]["profit_factor"] = 0.0
    ranked = [name for name, _ in rank_strategies(results, "profit_factor", 1)]
    assert ranked == ["FewTrades", "Winner", "Loser", "NoTrades"]


def _rank_args(testdatadir, *extra):
    return get_args(
        [
            "backtesting-rank",
            "--config",
            "config.json",
            "--datadir",
            str(testdatadir),
            "--strategy-path",
            str(Path(__file__).parents[1] / "strategy/strats"),
            "--hours",
            "240",
            *extra,
        ]
    )


def _patch_rank(mocker, default_conf):
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["ETH/BTC", "LTC/BTC"]),
    )
    default_conf["stake_amount"] = 0.01
    default_conf["max_open_trades"] = 2
    patched_configuration_load_config_file(mocker, default_conf)


def test_start_backtesting_rank(
    default_conf, mocker, testdatadir, time_machine, capsys, caplog, tmp_path
):
    # Test data ends 2018-01-30 04:50
    time_machine.move_to(datetime(2018, 1, 30, 5, 2, tzinfo=UTC), tick=False)
    _patch_rank(mocker, default_conf)
    dl_mock = mocker.patch("freqtrade.data.history.download_data_main")

    args = _rank_args(
        testdatadir,
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
        "StrategyTestV2",
        "HyperoptableStrategy",
        "TestStrategyNoImplements",
        "--top",
        "2",
        "--export-directory",
        str(tmp_path),
    )
    start_backtesting_rank(args)
    out = capsys.readouterr().out

    assert dl_mock.call_count == 0
    assert "STRATEGY RANKING - last 240 hours" in out
    assert "2018-01-20 05:00:00 -> 2018-01-30 04:55:00" in out
    for strategy in (CURRENT_TEST_STRATEGY, "StrategyTestV2", "HyperoptableStrategy"):
        assert strategy in out
    assert "Skipping strategy TestStrategyNoImplements" in caplog.text
    assert out.count("Total profit") == 2
    assert "#1 " in out
    assert "#2 " in out
    assert "#3 " not in out
    assert "TOTAL:" not in out
    assert "Tune them with: freqtrade hyperopt --strategy" in out
    assert "--timerange 20180120T0500-20180130T0455" in out
    reports = list(tmp_path.glob("backtest-rank-*.html"))
    assert len(reports) == 1
    assert f"HTML report written to {reports[0]}" in out
    html = reports[0].read_text(encoding="utf-8")
    assert "Strategy ranking" in html
    assert "#2 " in html


def _ranking(default_conf, mocker, testdatadir, time_machine) -> StrategyRanking:
    time_machine.move_to(datetime(2018, 1, 30, 5, 2, tzinfo=UTC), tick=False)
    _patch_rank(mocker, default_conf)
    mocker.patch(
        "freqtrade.optimize.strategy_ranking.recommend_stoploss",
        return_value=ParamSuggestion("stoploss", -0.10, -0.01, "test reason"),
    )
    mocker.patch("freqtrade.optimize.strategy_ranking.recommend_roi", return_value=None)
    mocker.patch("freqtrade.optimize.strategy_ranking.recommend_trailing", return_value=None)
    args = _rank_args(testdatadir, "--strategy-list", CURRENT_TEST_STRATEGY)
    return StrategyRanking(setup_optimize_configuration(args, RunMode.BACKTEST))


def test_strategy_ranking_recommend_not_improved(
    default_conf, mocker, testdatadir, time_machine, capsys
):
    ranking = _ranking(default_conf, mocker, testdatadir, time_machine)
    backtest_spy = mocker.spy(StrategyRanking, "_backtest_with_params")
    ranking.start()
    out = capsys.readouterr().out

    # A stoploss of -1% loses more on this data than the current -10%
    assert backtest_spy.call_count == 1
    assert backtest_spy.call_args[0][2] == {"stoploss": -0.01}
    assert backtest_spy.spy_return["stoploss"] == -0.01
    assert "RECOMMENDED PARAMETERS FOR StrategyTestV3" in out
    assert "test reason" in out
    assert "None of the derived settings improved the backtest" in out
    assert "Backtest with recommended parameters" not in out
    # Strategy is restored after the validation backtest
    strategy = ranking._get_strategy(CURRENT_TEST_STRATEGY)
    assert strategy.stoploss == -0.10
    assert strategy.config["stoploss"] == -0.10


def test_strategy_ranking_recommend_improved(
    default_conf, mocker, testdatadir, time_machine, capsys
):
    ranking = _ranking(default_conf, mocker, testdatadir, time_machine)
    orig = StrategyRanking._backtest_with_params

    def _better_backtest(self, name, params):
        stats = orig(self, name, params)
        return {**stats, "profit_total": 0.05}

    mocker.patch.object(StrategyRanking, "_backtest_with_params", _better_backtest)
    ranking.start()
    out = capsys.readouterr().out

    assert "Backtest with recommended parameters: profit 5.00%" in out
    assert "stoploss = -0.01" in out
    rec = ranking.recommend(CURRENT_TEST_STRATEGY)
    assert rec.applied == ["stoploss"]
    assert rec.recommended_params == {"stoploss": -0.01}
    assert rec.tuned_stats is not None
    assert rec.tuned_stats["profit_total"] == 0.05


def test_start_backtesting_rank_download(default_conf, mocker, testdatadir, time_machine):
    time_machine.move_to(datetime(2018, 1, 30, 5, 2, tzinfo=UTC), tick=False)
    _patch_rank(mocker, default_conf)
    dl_mock = mocker.patch("freqtrade.data.history.download_data_main")

    args = _rank_args(testdatadir, "--strategy-list", CURRENT_TEST_STRATEGY, "--download")
    start_backtesting_rank(args)

    assert dl_mock.call_count == 1
    dl_config = dl_mock.call_args[0][0]
    assert dl_config["timeframes"] == ["5m"]
    # 240 hours + 20 startup candles + 1
    assert dl_config["timerange"] == "20180120T0315-"
    assert dl_config["pairs"] == default_conf["exchange"]["pair_whitelist"]
    assert "prepend_data" not in dl_config


def test_start_backtesting_rank_download_prepend(default_conf, mocker, testdatadir, time_machine):
    # Local ETH/BTC and LTC/BTC data starts 2018-01-10 04:55 - 720 hours need older data
    time_machine.move_to(datetime(2018, 1, 30, 5, 2, tzinfo=UTC), tick=False)
    _patch_rank(mocker, default_conf)
    dl_mock = mocker.patch("freqtrade.data.history.download_data_main")

    args = _rank_args(testdatadir, "--strategy-list", CURRENT_TEST_STRATEGY, "--download")
    args["backtest_rank_hours"] = 720
    start_backtesting_rank(args)

    assert dl_mock.call_count == 2
    prepend_config = dl_mock.call_args_list[0][0][0]
    assert prepend_config["prepend_data"] is True
    # XRP/BTC and NEO/BTC have no local data - they are fully downloaded by the regular download
    assert prepend_config["pairs"] == ["ETH/BTC", "LTC/BTC"]
    assert prepend_config["timerange"] == "20171231T0315-"
    append_config = dl_mock.call_args_list[1][0][0]
    assert "prepend_data" not in append_config
    assert append_config["pairs"] == default_conf["exchange"]["pair_whitelist"]


def test_start_backtesting_rank_no_data(default_conf, mocker, testdatadir, time_machine, caplog):
    # No data available for this period
    time_machine.move_to(datetime(2024, 1, 1, tzinfo=UTC), tick=False)
    _patch_rank(mocker, default_conf)

    args = _rank_args(testdatadir, "--strategy-list", CURRENT_TEST_STRATEGY)
    with pytest.raises(OperationalException, match=r"No strategy could be backtested"):
        start_backtesting_rank(args)
    assert "Use --download to download the required data" in caplog.text


def test_start_backtesting_rank_no_strategies(default_conf, mocker, testdatadir, tmp_path):
    _patch_rank(mocker, default_conf)
    mocker.patch(
        "freqtrade.optimize.strategy_ranking.StrategyResolver.search_all_objects",
        MagicMock(return_value=[]),
    )
    args = _rank_args(testdatadir)
    with pytest.raises(OperationalException, match=r"No strategies found to rank"):
        start_backtesting_rank(args)
