"""
Strategy ranking - backtest all (or a list of) strategies over the last N hours,
show the best performing strategies and recommend stoploss / ROI / trailing settings for them.
"""

import logging
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np
from pandas import DataFrame
from rich.text import Text

from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.optimize.optimize_reports import generate_backtest_stats
from freqtrade.optimize.strategy_ranking_report import (
    build_line_chart,
    cumulative_profit_series,
    store_ranking_report,
)
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.util import dt_now, fmt_coin, print_rich_table


logger = logging.getLogger(__name__)

# Maps --rank-by choices to keys of the backtest statistics
RANK_METRICS: dict[str, str] = {
    "profit": "profit_total",
    "profit_factor": "profit_factor",
    "sharpe": "sharpe",
    "sortino": "sortino",
    "calmar": "calmar",
    "expectancy": "expectancy",
    "winrate": "winrate",
}

# Minimum amount of trades required before deriving parameters from the trade distribution
MIN_TRADES_FOR_RECOMMENDATION = 5
# Minimum amount of winning trades required to derive stoploss / trailing settings
MIN_WINNERS_FOR_RECOMMENDATION = 3

TRAILING_KEYS = (
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached",
)


@dataclass
class ParamSuggestion:
    name: str
    current: Any
    recommended: Any
    reason: str

    def as_strategy_params(self) -> dict[str, Any]:
        """Strategy attributes to set for this suggestion."""
        if self.name == "trailing":
            return dict(self.recommended)
        return {self.name: self.recommended}


@dataclass
class StrategyRecommendation:
    strategy: str
    suggestions: list[ParamSuggestion] = field(default_factory=list)
    # Names of the suggestions which improved the backtest result
    applied: list[str] = field(default_factory=list)
    tuned_stats: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def recommended_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for suggestion in self.suggestions:
            if suggestion.name in self.applied:
                params.update(suggestion.as_strategy_params())
        return params


@dataclass
class _BacktestGroup:
    """Strategies sharing one timeframe - backtested on the same data."""

    timeframe: str
    strategies: list[str]
    # Backtest window
    start: datetime
    end: datetime
    backtesting: Any = None
    data: dict[str, DataFrame] = field(default_factory=dict)
    min_date: datetime | None = None
    max_date: datetime | None = None


def get_lookback_window(
    hours: int, timeframe: str, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """
    Open dates of the first and the last candle of the last `hours` hours.
    The last candle is the last fully closed candle - both dates are inclusive.
    """
    current_candle = timeframe_to_prev_date(timeframe, now or dt_now())
    last_candle = current_candle - timedelta(minutes=timeframe_to_minutes(timeframe))
    return current_candle - timedelta(hours=hours), last_candle


def calculate_trade_excursions(trades: list[dict[str, Any]]) -> DataFrame:
    """
    Calculate max adverse (mae) and max favorable (mfe) excursion of each trade,
    as ratio of the open rate (leverage applied).
    mae is <= 0, mfe is >= 0.
    """
    columns = ["mae", "mfe", "profit_ratio", "trade_duration", "leverage"]
    if not trades:
        return DataFrame(columns=columns)
    df = DataFrame(trades)
    leverage = df["leverage"].fillna(1.0) if "leverage" in df else 1.0
    low = df["min_rate"].fillna(df["open_rate"]) / df["open_rate"] - 1
    high = df["max_rate"].fillna(df["open_rate"]) / df["open_rate"] - 1
    is_short = df["is_short"].astype(bool)
    mae = np.where(is_short, -high, low) * leverage
    mfe = np.where(is_short, -low, high) * leverage
    return DataFrame(
        {
            "mae": np.minimum(mae, 0.0),
            "mfe": np.maximum(mfe, 0.0),
            "profit_ratio": df["profit_ratio"].astype(float),
            "trade_duration": df["trade_duration"].astype(float),
            "leverage": leverage,
        },
        columns=columns,
    )


def _leverage(excursions: DataFrame) -> float:
    """Typical leverage of the trades - excursions and profits are multiplied by it."""
    if "leverage" not in excursions or excursions["leverage"].isna().all():
        return 1.0
    return float(excursions["leverage"].median())


def recommend_stoploss(excursions: DataFrame, current_stoploss: float) -> ParamSuggestion | None:
    """
    Place the stoploss just below the drawdown 90% of the winning trades went through.
    Winners keep running, while losing trades get cut earlier.
    """
    winners = excursions[excursions["profit_ratio"] > 0]
    if len(winners) < MIN_WINNERS_FOR_RECOMMENDATION:
        return None
    leverage = _leverage(excursions)
    depth = float(winners["mae"].abs().quantile(0.9))
    # At most a 35% price move - and always well before liquidation (-100% of the margin)
    stoploss = -round(
        min(max(depth * 1.2 + 0.005 * leverage, 0.01 * leverage), 0.35 * leverage, 0.8), 3
    )
    if abs(stoploss - current_stoploss) < 0.0025:
        return None
    losers_cut = int((excursions["profit_ratio"] < stoploss).sum())
    reason = f"90% of winning trades dipped less than {depth:.2%}"
    if losers_cut:
        reason += f"; {losers_cut} losing trades closed below {stoploss:.1%}"
    return ParamSuggestion("stoploss", current_stoploss, stoploss, reason)


def recommend_roi(
    excursions: DataFrame, current_roi: dict, timeframe: str, fee: float
) -> ParamSuggestion | None:
    """
    Build a ROI table targeting the move half of the trades reached (median favorable excursion),
    decaying over multiples of the median trade duration.
    """
    if len(excursions) < MIN_TRADES_FOR_RECOMMENDATION:
        return None
    min_profit = (2 * fee + 0.001) * _leverage(excursions)
    target = float(excursions["mfe"].median())
    if target <= min_profit * 2:
        return None
    step = max(int(excursions["trade_duration"].median()), timeframe_to_minutes(timeframe))
    roi = {
        "0": round(target, 4),
        str(step): round(max(target / 2, min_profit), 4),
        str(step * 2): round(max(target / 4, min_profit), 4),
        str(step * 4): 0,
    }
    current = {str(k): v for k, v in current_roi.items()}
    if roi == current:
        return None
    reason = f"half of the trades reached +{target:.2%}; median trade duration {step} min"
    return ParamSuggestion("minimal_roi", current, roi, reason)


def recommend_trailing(
    excursions: DataFrame, current: dict[str, Any], fee: float
) -> ParamSuggestion | None:
    """
    Recommend a trailing stop if winning trades give back most of their best unrealized profit.
    """
    winners = excursions[(excursions["profit_ratio"] > 0) & (excursions["mfe"] > 0)]
    if len(winners) < MIN_WINNERS_FOR_RECOMMENDATION:
        return None
    captured = float((winners["profit_ratio"] / winners["mfe"]).median())
    if captured >= 0.5:
        return None
    best_move = float(winners["mfe"].median())
    offset = round(best_move / 2, 4)
    positive = round(offset / 2, 4)
    if positive <= 2 * fee * _leverage(excursions):
        return None
    recommended = {
        "trailing_stop": True,
        "trailing_stop_positive": positive,
        "trailing_stop_positive_offset": offset,
        "trailing_only_offset_is_reached": True,
    }
    if recommended == current:
        return None
    reason = (
        f"winning trades kept only {captured:.0%} of their best profit "
        f"(median best move +{best_move:.2%})"
    )
    return ParamSuggestion("trailing", current, recommended, reason)


def get_metric(stats: dict[str, Any], key: str) -> float:
    """
    Metric value from backtest statistics - -inf if unavailable.
    """
    value = stats.get(key)
    if key == "profit_factor" and stats.get("losses", 0) == 0 and stats.get("wins", 0) > 0:
        # profit_factor is reported as 0 without losing trades
        return float("inf")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("-inf")
    return float(value)


def rank_strategies(
    results: dict[str, dict[str, Any]], rank_by: str, min_trades: int
) -> list[tuple[str, dict[str, Any]]]:
    """
    Sort strategy results, best first.
    Strategies with less than `min_trades` trades are always ranked last.
    """
    key = RANK_METRICS[rank_by]

    def _sort_key(item: tuple[str, dict[str, Any]]):
        stats = item[1]
        return (
            stats.get("total_trades", 0) >= min_trades,
            get_metric(stats, key),
            stats.get("profit_total", 0),
        )

    return sorted(results.items(), key=_sort_key, reverse=True)


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _result_label(profit: float) -> str:
    return "PROFIT" if profit > 0 else "LOSS" if profit < 0 else "-"


def _params_code(params: dict[str, Any]) -> str:
    return "\n".join(f"    {key} = {value!r}" for key, value in params.items())


def _profit_text(value: float, text: str) -> Text:
    return Text(text, style="green" if value > 0 else "red" if value < 0 else "")


def _fmt_ratio(value: float | None, decimals: int = 2) -> str:
    if value is None or np.isnan(value) or value == float("-inf"):
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.{decimals}f}"


class StrategyRanking:
    """
    Backtest strategies over the last N hours, rank them and recommend parameters
    for the best ones.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.hours: int = config.get("backtest_rank_hours", 168)
        self.top: int = config.get("backtest_rank_top", 3)
        self.rank_by: str = config.get("backtest_rank_by", "profit")
        self.min_trades: int = config.get("backtest_rank_min_trades", 1)
        self.groups: dict[str, _BacktestGroup] = {}
        # strategy name -> backtest statistics
        self.results: dict[str, dict[str, Any]] = {}
        self._strategy_group: dict[str, _BacktestGroup] = {}

    def get_strategy_names(self) -> list[str]:
        if self.config.get("strategy_list"):
            return list(dict.fromkeys(self.config["strategy_list"]))
        objs = StrategyResolver.search_all_objects(
            self.config, False, self.config.get("recursive_strategy_search", False)
        )
        return sorted({obj["name"] for obj in objs})

    def _group_strategies(self, names: list[str], now: datetime) -> None:
        """
        Load every strategy once to find its timeframe - and skip strategies which can't load.
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for name in names:
            strat_config = deepcopy(self.config)
            strat_config["strategy"] = name
            try:
                strategy = StrategyResolver.load_strategy(strat_config)
            except Exception as e:
                logger.warning(f"Skipping strategy {name}, it could not be loaded: {e}")
                continue
            groups[str(strategy.timeframe)].append(name)
        self.groups = {
            tf: _BacktestGroup(tf, strats, *get_lookback_window(self.hours, tf, now))
            for tf, strats in groups.items()
        }

    def _download_data(self, group: _BacktestGroup, group_config: Config) -> None:
        """
        Download the data required to backtest the group - including the startup candles.
        Data before already downloaded data is prepended, new data is appended.
        """
        from freqtrade.data.history import download_data_main, get_datahandler

        startup = max(
            StrategyResolver.load_strategy(
                {**deepcopy(group_config), "strategy": s}
            ).startup_candle_count
            for s in group.strategies
        )
        start = group.start - timedelta(
            minutes=timeframe_to_minutes(group.timeframe) * (startup + 1)
        )
        dl_config = deepcopy(group_config)
        dl_config.pop("days", None)
        dl_config["timerange"] = f"{start:%Y%m%dT%H%M}-"
        dl_config["timeframes"] = [group.timeframe]
        if group_config.get("timeframe_detail"):
            dl_config["timeframes"].append(group_config["timeframe_detail"])
        dl_config["pairs"] = group_config.get("pairs") or group_config["exchange"]["pair_whitelist"]
        dl_config.setdefault("new_pairs_days", 30)

        # A regular download only appends to existing data - prepend missing older candles first
        data_handler = get_datahandler(group_config["datadir"], group_config["dataformat_ohlcv"])
        candle_type = group_config.get("candle_type_def", CandleType.SPOT)
        prepend_pairs = []
        for pair in dl_config["pairs"]:
            for timeframe in dl_config["timeframes"]:
                first, _, candles = data_handler.ohlcv_data_min_max(pair, timeframe, candle_type)
                if candles and first > start:
                    prepend_pairs.append(pair)
                    break
        if prepend_pairs:
            logger.info(f"Prepending data since {start} for {', '.join(prepend_pairs)} ...")
            download_data_main(
                {**deepcopy(dl_config), "pairs": prepend_pairs, "prepend_data": True}
            )

        logger.info(f"Downloading {', '.join(dl_config['timeframes'])} data since {start} ...")
        download_data_main(dl_config)

    def _run_group(self, group: _BacktestGroup) -> None:
        from freqtrade.optimize.backtesting import Backtesting

        group_config = deepcopy(self.config)
        group_config["timeframe"] = group.timeframe
        group_config["strategy"] = group.strategies[0]
        group_config["strategy_list"] = group.strategies
        group_config["export"] = "none"
        group_config["timerange"] = f"{group.start:%Y%m%dT%H%M}-{group.end:%Y%m%dT%H%M}"

        if self.config.get("backtest_rank_download"):
            self._download_data(group, group_config)

        bt = Backtesting(group_config)
        group.backtesting = bt
        with bt.progress or nullcontext():
            group.data, timerange = bt.load_bt_data()
            for strat in bt.strategylist:
                name = strat.get_strategy_name()
                try:
                    group.min_date, group.max_date = bt.backtest_one_strategy(
                        strat, group.data, timerange
                    )
                except Exception as e:
                    logger.warning(f"Backtesting {name} failed, skipping it: {e}")
                    bt.all_bt_content.pop(name, None)
        # Progress bars are only wanted for the main run
        bt.progress = None

        if not bt.all_bt_content or group.min_date is None or group.max_date is None:
            return
        stats = generate_backtest_stats(
            group.data, bt.all_bt_content, min_date=group.min_date, max_date=group.max_date
        )
        for name, strat_stats in stats["strategy"].items():
            self.results[name] = strat_stats
            self._strategy_group[name] = group

    def _get_strategy(self, name: str) -> IStrategy:
        group = self._strategy_group[name]
        return next(s for s in group.backtesting.strategylist if s.get_strategy_name() == name)

    def _backtest_with_params(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Re-run the backtest of one strategy with the given strategy attributes overridden.
        The strategy is restored afterwards.
        """
        group = self._strategy_group[name]
        if group.min_date is None or group.max_date is None:
            raise OperationalException(f"No backtest result available for {name}.")
        min_date, max_date = group.min_date, group.max_date
        bt = group.backtesting
        strategy = self._get_strategy(name)
        previous = {key: getattr(strategy, key, None) for key in params}
        previous_config = {key: strategy.config.get(key) for key in params}

        def _set(values: dict[str, Any]) -> None:
            for key, value in values.items():
                if key == "minimal_roi" and value is not None:
                    value = dict(sorted({int(k): v for k, v in value.items()}.items()))
                setattr(strategy, key, value)

        try:
            _set(params)
            strategy.config.update(params)
            bt.backtest_one_strategy(strategy, group.data, bt.timerange)
            stats = generate_backtest_stats(
                group.data,
                {name: bt.all_bt_content[name]},
                min_date=min_date,
                max_date=max_date,
            )
            return stats["strategy"][name]
        finally:
            _set(previous)
            strategy.config.update(previous_config)

    def recommend(self, name: str) -> StrategyRecommendation:
        """
        Derive parameter suggestions from the trades of a strategy, and keep the combination
        of suggestions which improves the backtest result the most.
        """
        stats = self.results[name]
        strategy = self._get_strategy(name)
        rec = StrategyRecommendation(name)
        excursions = calculate_trade_excursions(stats["trades"])
        if len(excursions) < MIN_TRADES_FOR_RECOMMENDATION:
            rec.notes.append(
                f"Only {len(excursions)} trades - not enough to derive parameters. "
                f"Use a longer period (--hours)."
            )
            return rec

        fee = self._strategy_group[name].backtesting.fee
        current_trailing = {key: getattr(strategy, key, None) for key in TRAILING_KEYS}
        suggestions = [
            recommend_stoploss(excursions, strategy.stoploss),
            recommend_roi(excursions, strategy.minimal_roi, str(strategy.timeframe), fee),
            recommend_trailing(excursions, current_trailing, fee),
        ]
        rec.suggestions = [s for s in suggestions if s is not None]
        if strategy.use_custom_stoploss:
            rec.notes.append(
                "Strategy uses custom_stoploss() - the stoploss acts as hard limit only."
            )
        if not rec.suggestions:
            rec.notes.append(
                "The trades give no reason to change stoploss / ROI / trailing settings."
            )
            return rec

        metric_key = RANK_METRICS[self.rank_by]
        best_value = get_metric(stats, metric_key)
        for size in range(1, len(rec.suggestions) + 1):
            for combo in combinations(rec.suggestions, size):
                params: dict[str, Any] = {}
                for suggestion in combo:
                    params.update(suggestion.as_strategy_params())
                tuned = self._backtest_with_params(name, params)
                value = get_metric(tuned, metric_key)
                if tuned.get("total_trades", 0) >= self.min_trades and value > best_value:
                    best_value = value
                    rec.applied = [s.name for s in combo]
                    rec.tuned_stats = tuned
        if not rec.applied:
            rec.notes.append(
                "None of the derived settings improved the backtest - keep current settings."
            )
        return rec

    def show_ranking(self, ranked: list[tuple[str, dict[str, Any]]]) -> None:
        stake_currency = self.config["stake_currency"]
        metric_key = RANK_METRICS[self.rank_by]
        rows = []
        for idx, (name, stats) in enumerate(ranked, start=1):
            profit = stats["profit_total"]
            row: list[str | Text] = [
                str(idx),
                name,
                stats["timeframe"],
                str(stats["total_trades"]),
                f"{stats['wins']}/{stats['draws']}/{stats['losses']}",
                f"{stats['winrate']:.1%}",
                _profit_text(profit, f"{profit:.2%}"),
                _profit_text(profit, fmt_coin(stats["profit_total_abs"], stake_currency)),
                _profit_text(profit, _result_label(profit)),
                f"{stats.get('max_drawdown_account', 0.0):.2%}",
            ]
            if self.rank_by != "profit":
                row.append(_fmt_ratio(get_metric(stats, metric_key)))
            rows.append(row)
        headers = [
            "Rank",
            "Strategy",
            "TF",
            "Trades",
            "Win/Draw/Loss",
            "Win%",
            "Profit %",
            f"Profit {stake_currency}",
            "Result",
            "Max DD",
        ]
        if self.rank_by != "profit":
            headers.append(self.rank_by)
        start = min(g.start for g in self.groups.values())
        end = max(g.end for g in self.groups.values())
        print_rich_table(
            rows,
            headers,
            summary=(
                f"STRATEGY RANKING - last {self.hours} hours ("
                f"{start.strftime(DATETIME_PRINT_FORMAT)} -> "
                f"{end.strftime(DATETIME_PRINT_FORMAT)}) by {self.rank_by}"
            ),
        )

    def _detail_rows(self, stats: dict[str, Any]) -> list[tuple[str, str, int]]:
        """
        Key metrics of one strategy as (label, value, sign of the value) rows.
        """
        stake_currency = self.config["stake_currency"]
        profit = stats["profit_total"]
        best_pair = stats.get("best_pair") or {}
        worst_pair = stats.get("worst_pair") or {}
        exit_reasons = sorted(
            (x for x in stats.get("exit_reason_summary", []) if x["key"] != "TOTAL"),
            key=lambda x: x["trades"],
            reverse=True,
        )
        rows = [
            (
                "Total profit",
                f"{fmt_coin(stats['profit_total_abs'], stake_currency)} ({profit:.2%})",
                _sign(profit),
            ),
            (
                "Balance",
                (
                    f"{fmt_coin(stats['starting_balance'], stake_currency)} -> "
                    f"{fmt_coin(stats['final_balance'], stake_currency)}"
                ),
                0,
            ),
            ("Trades", f"{stats['total_trades']} ({stats['trades_per_day']} / day)", 0),
            (
                "Win / Draw / Loss",
                (
                    f"{stats['wins']} / {stats['draws']} / {stats['losses']} "
                    f"(win rate {stats['winrate']:.1%})"
                ),
                0,
            ),
            ("Avg. profit per trade", f"{stats['profit_mean']:.2%}", _sign(stats["profit_mean"])),
            ("Profit factor", _fmt_ratio(get_metric(stats, "profit_factor")), 0),
            ("Expectancy (ratio)", _fmt_ratio(stats.get("expectancy_ratio")), 0),
            ("Max drawdown", f"{stats.get('max_drawdown_account', 0.0):.2%}", 0),
            (
                "Sharpe / Sortino",
                f"{_fmt_ratio(stats.get('sharpe'))} / {_fmt_ratio(stats.get('sortino'))}",
                0,
            ),
            ("Avg. trade duration", str(stats.get("holding_avg", "-")), 0),
            ("Market change", f"{stats.get('market_change', 0.0):.2%}", 0),
        ]
        if best_pair:
            rows.append(
                (
                    "Best pair",
                    f"{best_pair['key']} {best_pair['profit_total']:.2%}",
                    _sign(best_pair["profit_total"]),
                )
            )
        if worst_pair:
            rows.append(
                (
                    "Worst pair",
                    f"{worst_pair['key']} {worst_pair['profit_total']:.2%}",
                    _sign(worst_pair["profit_total"]),
                )
            )
        if exit_reasons:
            rows.append(
                (
                    "Exit reasons",
                    ", ".join(f"{x['key']}: {x['trades']}" for x in exit_reasons[:4]),
                    0,
                )
            )
        return rows

    def show_strategy_details(self, rank: int, name: str, stats: dict[str, Any]) -> None:
        profit = stats["profit_total"]
        rows: list[tuple[str, str | Text]] = [
            ("Result", _profit_text(profit, _result_label(profit)))
        ]
        rows.extend(
            (label, _profit_text(sign, value) if sign else value)
            for label, value, sign in self._detail_rows(stats)
        )
        print_rich_table(
            rows,
            ["Metric", "Value"],
            summary=f"#{rank} {name} ({stats['timeframe']}, "
            f"{stats['backtest_start']} -> {stats['backtest_end']})",
            justify="left",
        )

    def _hyperopt_hint(self, name: str, stats: dict[str, Any]) -> tuple[str, str] | None:
        """
        Current hyperoptable parameter values and a hyperopt command for the same window.
        """
        hyper_params = list(self._get_strategy(name).enumerate_parameters())
        if not hyper_params:
            return None
        values = ", ".join(f"{param_name}={param.value}" for param_name, param in hyper_params)
        command = (
            f"freqtrade hyperopt --strategy {name} "
            f"--timeframe {stats['timeframe']} --timerange {stats['timerange']} "
            "--spaces buy sell roi stoploss trailing "
            "--hyperopt-loss SharpeHyperOptLossDaily --epochs 200"
        )
        return values, command

    def _tuned_summary(self, rec: StrategyRecommendation, stats: dict[str, Any]) -> str | None:
        if not rec.tuned_stats:
            return None
        tuned = rec.tuned_stats
        return (
            f"Backtest with recommended parameters: profit {tuned['profit_total']:.2%} "
            f"(was {stats['profit_total']:.2%}), {tuned['total_trades']} trades, "
            f"win rate {tuned['winrate']:.1%}, "
            f"max drawdown {tuned.get('max_drawdown_account', 0.0):.2%}"
        )

    def _notes(self, rec: StrategyRecommendation) -> list[str]:
        notes = list(rec.notes)
        if rec.applied:
            notes.append(
                f"Recommendations are fitted to the last {self.hours} hours only - "
                "verify them on a different period before using them live."
            )
        return notes

    def show_recommendation(self, rec: StrategyRecommendation, stats: dict[str, Any]) -> None:
        if rec.suggestions:
            rows: list[list[str | Text]] = []
            for s in rec.suggestions:
                applied = s.name in rec.applied
                rows.append(
                    [
                        s.name,
                        str(s.current),
                        Text(str(s.recommended), style="green" if applied else "dim"),
                        "yes" if applied else "no",
                        s.reason,
                    ]
                )
            print_rich_table(
                rows,
                ["Parameter", "Current", "Suggested", "Recommended", "Reason"],
                summary=f"RECOMMENDED PARAMETERS FOR {rec.strategy}",
                justify="left",
            )
        tuned_summary = self._tuned_summary(rec, stats)
        if tuned_summary:
            print(tuned_summary)
            print("Add to the strategy (or config):")
            print(_params_code(rec.recommended_params))

        hint = self._hyperopt_hint(rec.strategy, stats)
        if hint:
            print(f"Current hyperoptable parameters: {hint[0]}")
            print(f"Tune them with: {hint[1]}")
        for note in self._notes(rec):
            print(f"Note: {note}")
        print()

    def start(self) -> None:
        names = self.get_strategy_names()
        if not names:
            raise OperationalException("No strategies found to rank.")
        logger.info(f"Ranking {len(names)} strategies over the last {self.hours} hours.")
        self._group_strategies(names, dt_now())

        for group in self.groups.values():
            try:
                self._run_group(group)
            except OperationalException as e:
                logger.warning(
                    f"Backtesting {group.timeframe} strategies {group.strategies} failed: {e}. "
                    "Use --download to download the required data."
                )

        if not self.results:
            raise OperationalException(
                "No strategy could be backtested. "
                "Make sure data is available for the selected period (see --download)."
            )

        ranked = rank_strategies(self.results, self.rank_by, self.min_trades)
        self.show_ranking(ranked)

        top = [item for item in ranked if item[1]["total_trades"] >= self.min_trades][: self.top]
        if not top:
            logger.warning(f"No strategy reached the minimum of {self.min_trades} trades.")
        recommendations = []
        for rank, (name, stats) in enumerate(top, start=1):
            self.show_strategy_details(rank, name, stats)
            rec = self.recommend(name)
            self.show_recommendation(rec, stats)
            recommendations.append(rec)

        report = store_ranking_report(self.config, self.report_data(ranked, recommendations))
        print(f"HTML report written to {report}")

    def report_data(
        self,
        ranked: list[tuple[str, dict[str, Any]]],
        recommendations: list[StrategyRecommendation],
    ) -> dict[str, Any]:
        """
        Data for the HTML report.
        """
        stake_currency = self.config["stake_currency"]
        metric_key = RANK_METRICS[self.rank_by]
        start = min(g.start for g in self.groups.values())
        end = max(g.end for g in self.groups.values())

        ranking = [
            {
                "rank": idx,
                "name": name,
                "timeframe": stats["timeframe"],
                "trades": stats["total_trades"],
                "wdl": f"{stats['wins']} / {stats['draws']} / {stats['losses']}",
                "winrate": f"{stats['winrate']:.1%}",
                "profit_total": stats["profit_total"],
                "profit_pct": f"{stats['profit_total']:.2%}",
                "profit_abs": fmt_coin(stats["profit_total_abs"], stake_currency),
                "sign": _sign(stats["profit_total"]),
                "result": _result_label(stats["profit_total"]),
                "max_dd": f"{stats.get('max_drawdown_account', 0.0):.2%}",
                "metric": _fmt_ratio(get_metric(stats, metric_key)),
            }
            for idx, (name, stats) in enumerate(ranked, start=1)
        ]

        strategies = []
        for rank, rec in enumerate(recommendations, start=1):
            stats = self.results[rec.strategy]
            group = self._strategy_group[rec.strategy]
            if group.min_date is None or group.max_date is None:
                continue
            series = [
                {
                    "name": "Current parameters",
                    "slot": 1,
                    "points": cumulative_profit_series(
                        stats["trades"], group.min_date, group.max_date
                    ),
                }
            ]
            if rec.tuned_stats:
                series.append(
                    {
                        "name": "Recommended parameters",
                        "slot": 2,
                        "points": cumulative_profit_series(
                            rec.tuned_stats["trades"], group.min_date, group.max_date
                        ),
                    }
                )
            hint = self._hyperopt_hint(rec.strategy, stats)
            strategies.append(
                {
                    "rank": rank,
                    "name": rec.strategy,
                    "timeframe": stats["timeframe"],
                    "period": f"{stats['backtest_start']} -> {stats['backtest_end']}",
                    "sign": _sign(stats["profit_total"]),
                    "result": _result_label(stats["profit_total"]),
                    "details": [
                        {"label": label, "value": value, "sign": sign}
                        for label, value, sign in self._detail_rows(stats)
                    ],
                    "chart": build_line_chart(series, group.min_date, group.max_date),
                    "suggestions": [
                        {
                            "name": sug.name,
                            "current": str(sug.current),
                            "recommended": str(sug.recommended),
                            "applied": sug.name in rec.applied,
                            "reason": sug.reason,
                        }
                        for sug in rec.suggestions
                    ],
                    "tuned": self._tuned_summary(rec, stats),
                    "params_code": _params_code(rec.recommended_params) if rec.applied else None,
                    "hyperopt_params": hint[0] if hint else None,
                    "hyperopt_cmd": hint[1] if hint else None,
                    "notes": self._notes(rec),
                }
            )

        profitable = sum(1 for _, stats in ranked if stats["profit_total"] > 0)
        best_name, best_stats = ranked[0]
        market_change = best_stats.get("market_change", 0.0)
        tiles = [
            {
                "label": "Best strategy",
                "value": best_name,
                "note": f"{best_stats['profit_total']:.2%} · "
                f"{fmt_coin(best_stats['profit_total_abs'], stake_currency)}",
                "sign": 0,
            },
            {
                "label": "Profitable strategies",
                "value": f"{profitable} of {len(ranked)}",
                "note": f"over the last {self.hours} hours",
                "sign": 0,
            },
            {
                "label": "Market change",
                "value": f"{market_change:.2%}",
                "note": "average change of all pairs",
                "sign": _sign(market_change),
            },
        ]
        pairs = sorted({pair for _, stats in ranked for pair in stats.get("pairlist", [])})
        return {
            "hours": self.hours,
            "window_start": start.strftime(DATETIME_PRINT_FORMAT),
            "window_end": end.strftime(DATETIME_PRINT_FORMAT),
            "rank_by": self.rank_by,
            "stake_currency": stake_currency,
            "pairs": pairs,
            "tiles": tiles,
            "metric_header": self.rank_by if self.rank_by != "profit" else None,
            "ranking": ranking,
            "strategies": strategies,
        }
