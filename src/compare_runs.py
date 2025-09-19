import json
import os
import pandas as pd
import argparse


def load_run_results(directory, pattern="*.json"):
    """Load all JSON files in a dir matching pattern, extract key fields."""
    # Convert relative path to absolute path from current working directory
    abs_dir = os.path.abspath(directory)
    print(f"DEBUG: Looking for directory: {abs_dir}")
    print(f"DEBUG: Directory exists: {os.path.exists(abs_dir)}")

    if not os.path.exists(abs_dir):
        print(f"ERROR: Directory not found: {abs_dir}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        return []

    # List all files to debug
    all_files = os.listdir(abs_dir)
    print(f"DEBUG: Files in directory: {all_files}")

    json_files = [f for f in all_files if f.endswith(".json")]
    print(f"DEBUG: JSON files found: {json_files}")

    if not json_files:
        print(f"No JSON files found in {abs_dir}")
        return []

    results = []
    for filename in json_files:
        filepath = os.path.join(abs_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Extract your three things
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
            print(f"DEBUG: Successfully loaded {filename}")
        except Exception as e:
            print(f"ERROR loading {filename}: {e}")
            continue

    return results


def compare_runs(results_list, metric_keys=None, param_keys=None):
    """Create comparison tables."""
    if not results_list:
        print("No results to compare!")
        return

    if not metric_keys:
        metric_keys = ["accuracy", "f1", "roc_auc", "recall"]  # Defaults
    if not param_keys:
        param_keys = (
            list(results_list[0]["params"].keys()) if results_list else []
        )  # Auto-detect

    # Metrics table
    metrics_data = []
    for r in results_list:
        row = {"run": r["run_file"][:25]}  # Truncate filename
        for k in metric_keys:
            row[k] = r["metrics"].get(k, "N/A")
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    print("\n=== METRICS COMPARISON ===")
    print(metrics_df.round(4).to_string(index=False))

    # Params table
    if param_keys:
        params_data = []
        for r in results_list:
            row = {"run": r["run_file"][:25]}
            for k in param_keys:
                row[k] = r["params"].get(k, "N/A")
            params_data.append(row)

        params_df = pd.DataFrame(params_data)
        print("\n=== PARAMS COMPARISON ===")
        print(params_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="evaluation_results", help="Dir with JSON files"
    )
    args = parser.parse_args()

    print(f"Starting comparison with directory: {args.dir}")
    results = load_run_results(args.dir)
    if results:
        print(f"\n✅ Loaded {len(results)} runs from {args.dir}")
        compare_runs(results)
    else:
        print("\n❌ No results found!")
