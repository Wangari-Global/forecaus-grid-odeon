"""Feature engineering: calendar encodings + lag/rolling modelling frame.

National frame: :func:`build` / :func:`train_test_split`.
Per-substation (LV-feeder) frames for federation: :mod:`.ss`.
"""
from .build_features import DEFAULT_LAGS, DEFAULT_ROLLING, build, train_test_split
from .calendar import add_calendar
from .ss import adapt_lags, build_ss_frame, periods_per_day, ss_train_test_split

__all__ = [
    "add_calendar",
    "build",
    "train_test_split",
    "DEFAULT_LAGS",
    "DEFAULT_ROLLING",
    "build_ss_frame",
    "ss_train_test_split",
    "adapt_lags",
    "periods_per_day",
]
