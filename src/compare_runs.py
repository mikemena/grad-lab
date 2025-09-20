import json
import os
import argparse
from tabulate import tabulate
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def load_run_results(directory, pattern="*.json"):
    """Load all JSON files in a dir matching pattern, extract key fields."""
    # Convert relative path to absolute path from current working directory
    abs_dir = os.path.abspath(directory)
    logger.info(f"DEBUG: Looking for directory: {abs_dir}")
    logger.info(f"DEBUG: Directory exists: {os.path.exists(abs_dir)}")

    if not os.path.exists(abs_dir):
        logger.info(f"ERROR: Directory not found: {abs_dir}")
        logger.info(f"DEBUG: Current working directory: {os.getcwd()}")
        return []

    # List all files to debug
    all_files = os.listdir(abs_dir)
    logger.info(f"DEBUG: Files in directory: {all_files}")

    json_files = [f for f in all_files if f.endswith(".json")]
    logger.info(f"DEBUG: JSON files found: {json_files}")

    if not json_files:
        logger.error(f"No JSON files found in {abs_dir}")
        return []

    results = []
    for filename in json_files:
        filepath = os.path.join(abs_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Extract the components
            run_date = data.get("date", "Unknown")
            params = data.get("full_config", data.get("model_config", {}))  # Try both
            metrics = data.get(
                "test_metrics", data.get("metrics", {})
            )  # Flexible for both files

            # Simple param extraction for now
            flat_params = {"sample_param": "N/A"}  # Placeholder
            if isinstance(params, dict):
                # Extract any top-level numeric/string params
                flat_params = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str)) and k not in ["type"]
                }

            results.append(
                {
                    "run_file": filename,
                    "date": run_date,
                    "params": flat_params,
                    "metrics": metrics,
                }
            )
            logger.info(f"DEBUG: Successfully loaded {filename}")
        except Exception as e:
            logger.error(f"ERROR loading {filename}: {e}")
            continue

    return results


def compare_runs(results_list, metric_keys=None, param_keys=None):
    """Create comparison tables."""
    if not results_list:
        logger.warning("No results to compare!")
        return

    if not metric_keys:
        metric_keys = [
            "accuracy",
            "f1",
            "roc_auc",
            "recall",
            "matthews_corrcoef",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ]  # Defaults
    if not param_keys:
        param_keys = (
            list(results_list[0]["params"].keys()) if results_list else []
        )  # Auto-detect

    # Metrics table with tabulate
    metrics_data = []
    for r in results_list:
        row = [r["run_file"][:35]]  # First column
        for k in metric_keys:
            val = r["metrics"].get(k, "N/A")
            if isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        metrics_data.append(row)

    headers = ["run"] + metric_keys

    logger.info("\n=== METRICS COMPARISON ===")
    logger.info(
        tabulate(metrics_data, headers=headers, tablefmt="grid", floatfmt=".4f")
    )

    # Params table
    if param_keys:
        params_data = []
        for r in results_list:
            row = {"run": r["run_file"][:25]}  # First column
            for k in param_keys:
                row[k] = r["params"].get(k, "N/A")
            params_data.append(row)

        logger.info("\n=== PARAMS COMPARISON ===")
        logger.info(
            tabulate(params_data, headers=["run"] + param_keys, tablefmt="grid")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="evaluation_results", help="Dir with JSON files"
    )
    args = parser.parse_args()

    logger.info(f"Starting comparison with directory: {args.dir}")
    results = load_run_results(args.dir)
    if results:
        logger.info(f"\n✅ Loaded {len(results)} runs from {args.dir}")
        compare_runs(results)
    else:
        logger.error("\n❌ No results found!")
