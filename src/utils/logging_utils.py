import json
import logging
import datetime
from pathlib import Path

from src.utils.config import EXP_LOGS_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger with a consistent format.

    Usage in any phase script:
        from src.utils.logging_utils import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_experiment(
    run_name: str,
    model_name: str,
    feature_set_label: str,
    feature_list: list,
    random_seed: int,
    metrics: dict,
    figure_paths: list = None,
    extra: dict = None,
):
    """
    Writes a single JSON record to logs/experiments/ for one experiment run.
    """
    EXP_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "run_name":          run_name,
        "timestamp":         datetime.datetime.now().isoformat(),
        "model_name":        model_name,
        "feature_set_label": feature_set_label,
        "feature_list":      feature_list,
        "n_features":        len(feature_list),
        "random_seed":       random_seed,
        "metrics":           metrics,
        "figure_paths":      [str(p) for p in (figure_paths or [])],
    }

    if extra:
        record.update(extra)

    log_path = EXP_LOGS_DIR / f"{run_name}.json"
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Log saved: {log_path}")
    return log_path


def _fmt(val):
    """Format a metric value that may be float or None."""
    if val is None:
        return "  n/a  "
    return f"{val:.4f}"


def print_metrics_summary(run_name: str, metrics: dict):
    """Quick terminal summary after a run."""
    print(f"\n{'='*55}")
    print(f"  {run_name}")
    print(f"{'='*55}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<30} {_fmt(val)}")
        elif isinstance(val, dict):
            mean  = _fmt(val.get("mean"))
            lower = _fmt(val.get("lower"))
            upper = _fmt(val.get("upper"))
            print(f"  {key:<30} {mean}  [{lower}, {upper}]")
        else:
            print(f"  {key:<30} {val}")
    print()
