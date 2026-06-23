"""Evaluation: forecast metrics + rolling-origin backtest + the headline
causal-vs-correlational structural-break experiment."""
from .backtest import format_table, rolling_origin
from .causal_break import make_break_figure, run_break_experiment
from .metrics import coverage, mae, mape, rmse, summary
from .ss_benchmark import aggregate_tables, run_ss_benchmark
from .ss_causal import (generate_ss_break_data, run_ss_break_experiment,
                        run_ss_causal_effect)

__all__ = [
    "mae", "rmse", "mape", "coverage", "summary",
    "rolling_origin", "format_table",
    "run_break_experiment", "make_break_figure",
    "run_ss_benchmark", "aggregate_tables",
    "run_ss_causal_effect", "run_ss_break_experiment", "generate_ss_break_data",
]
