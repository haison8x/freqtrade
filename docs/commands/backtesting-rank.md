``` output
usage: freqtrade backtesting-rank [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                  [-c PATH] [-d PATH] [--userdir PATH]
                                  [-s NAME] [--strategy-path PATH]
                                  [--recursive-strategy-search]
                                  [--freqaimodel NAME]
                                  [--freqaimodel-path PATH] [-i TIMEFRAME]
                                  [--data-format-ohlcv {json,jsongz,feather,parquet}]
                                  [--max-open-trades INT]
                                  [--stake-amount STAKE_AMOUNT] [--fee FLOAT]
                                  [-p PAIRS [PAIRS ...]]
                                  [--enable-protections]
                                  [--dry-run-wallet DRY_RUN_WALLET]
                                  [--timeframe-detail TIMEFRAME_DETAIL]
                                  [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]
                                  [--hours HOURS] [--top INT]
                                  [--rank-by {profit,profit_factor,sharpe,sortino,calmar,expectancy,winrate}]
                                  [--min-trades INT] [--download]
                                  [--backtest-directory PATH]

options:
  -h, --help            show this help message and exit
  -i, --timeframe TIMEFRAME
                        Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `feather`).
  --max-open-trades INT
                        Override the value of the `max_open_trades`
                        configuration setting.
  --stake-amount STAKE_AMOUNT
                        Override the value of the `stake_amount` configuration
                        setting.
  --fee FLOAT           Specify fee ratio. Will be applied twice (on trade
                        entry and exit).
  -p, --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --enable-protections, --enableprotections
                        Enable protections for backtesting. Will slow
                        backtesting down by a considerable amount, but will
                        include configured protections
  --dry-run-wallet, --starting-balance DRY_RUN_WALLET
                        Starting balance, used for backtesting / hyperopt and
                        dry-runs.
  --timeframe-detail TIMEFRAME_DETAIL
                        Specify detail timeframe for backtesting (`1m`, `5m`,
                        `30m`, `1h`, `1d`).
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        Provide a space-separated list of strategies to rank.
                        Defaults to all strategies in the strategy directory.
  --hours HOURS         Backtest the last HOURS hours, ending at the last
                        closed candle (default: 168).
  --top INT             Show details and recommended parameters for the best N
                        strategies (default: 3).
  --rank-by {profit,profit_factor,sharpe,sortino,calmar,expectancy,winrate}
                        Metric used to rank strategies (default: profit).
  --min-trades INT      Minimal number of trades for a strategy to be ranked
                        among the best (default: 1).
  --download            Download / update the required candle data before
                        backtesting.
  --backtest-directory, --export-directory PATH
                        Directory to use for backtest results. Example:
                        `--export-directory=user_data/backtest_results/`.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.
  --logfile, --log-file FILE
                        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c, --config PATH     Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  -d, --datadir, --data-dir PATH
                        Path to the base directory of the exchange with
                        historical backtesting data. To see futures data, use
                        trading-mode additionally.
  --userdir, --user-data-dir PATH
                        Path to userdata directory.

Strategy arguments:
  -s, --strategy NAME   Specify strategy class name which will be used by the
                        bot.
  --strategy-path PATH  Specify additional strategy lookup path.
  --recursive-strategy-search
                        Recursively search for a strategy in the strategies
                        folder.
  --freqaimodel NAME    Specify a custom freqaimodels.
  --freqaimodel-path PATH
                        Specify additional lookup path for freqaimodels.

```
