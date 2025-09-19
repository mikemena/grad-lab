import os
import json
import pandas as pd
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def load_run_results(directory, pattern="*.json"):
    """Load all JSON files in a dir matching pattern, extract key fields"""
    results = []
    for filename in os.listdir(directory):
        if pattern in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            # Extract date, params, metrics
            run_date = data.get("date", "Unknown")
            params = data.get("full_config", {})
            metrics = data.get("test_metrics", data.get("metrics", {}))

            # Flatten params for table
            flat_params = {
                k: v for k, v in params.items() if isinstance(v, (int, float, str))
            }

            results.append(
                {
                    "run_file": filename,
                    "date": run_date,
                    "params": flat_params,
                    "metrics": metrics,
                }
            )
    return results


def compare_runs(results_list, metric_keys=None, param_keys=None):
    """Create comparison tables"""
    if not metric_keys:
        metric_keys = ["accuracy", "f1", "roc_auc", "recall"]
    if not param_keys:
        param_keys = ["lr", "hidden_dims", "dropout_rate"]

    # Metrics table
    metrics_df = pd.DataFrame(
        [
            {
                "run": r["run_file"],
                **{k: r["metrics"].get(k, "N/A") for k in metric_keys},
            }
            for r in results_list
        ]
    )
    logger.info("--- COMPARE METRICS ---")
    logger.info(metrics_df.round(4).to_string(index=False))

    # Params table
    params_df = pd.DataFrame(
        [
            {
                "run": r["run_file"],
                **{k: r["params"].get(k, "N/A") for k in param_keys},
            }
            for r in results_list
        ]
    )
    logger.info("--- COMPARE PARAMS (CHANGES HIGHLIGHTED) ---")
    logger.info(params_df.to_string(index=False))
