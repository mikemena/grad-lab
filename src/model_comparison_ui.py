import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_all_runs(directory="evaluation_results"):
    """Load all JSON files and extract key data for UI."""
    abs_dir = os.path.abspath(directory)
    if not os.path.exists(abs_dir):
        st.error(f"Directory not found: {abs_dir}")
        return []

    json_files = [f for f in os.listdir(abs_dir) if f.endswith(".json")]
    runs = []

    for filename in json_files:
        filepath = os.path.join(abs_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract key info
            run_info = {
                "filename": filename,
                "date": data.get("date", "Unknown"),
                "metrics": data.get("metrics", data.get("test_metrics", {})),
                "params": data.get("params", data.get("params", {})),
                "predictions": data.get("predictions", []),
                "targets": data.get("targets", []),
            }

            # Extract key metrics
            run_info["key_metrics"] = {
                "accuracy": run_info["metrics"].get("accuracy", 0),
                "precision": run_info["metrics"].get("precision", 0),
                "recall": run_info["metrics"].get("recall", 0),
                "f1": run_info["metrics"].get("f1", 0),
                "roc_auc": run_info["metrics"].get("roc_auc", 0),
                "business_value": run_info["metrics"].get("total_business_value", 0),
            }

            # Extract key params (flattened)
            run_info["key_params"] = extract_key_params(run_info["params"])

            runs.append(run_info)

        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
            continue

    # Sort by date (newest first)
    runs.sort(key=lambda x: x["date"], reverse=True)
    return runs


def extract_key_params(params):
    """Extract important parameters for display."""
    key_params = {}
    if isinstance(params, dict):
        # Common ML params
        param_paths = {
            "lr": ["training", "lr"],
            "batch_size": ["training", "batch_size"],
            "epochs": ["training", "epochs"],
            "hidden_dims": ["model", "hidden_dims"],
            "dropout": ["model", "dropout_rate"],
            "model_type": ["model", "type"],
            "loss_type": ["training", "loss_type"],
            "optimizer": ["training", "optimizer_name"],
        }

        for param_name, path in param_paths.items():
            value = params
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value is not None:
                key_params[param_name] = value

        # Add top-level params
        for k, v in params.items():
            if k not in ["training", "model"] and isinstance(v, (str, int, float)):
                key_params[k] = v

    return key_params


def create_comparison_df(selected_runs):
    """Create DataFrame for side-by-side comparison."""
    if len(selected_runs) < 2:
        return pd.DataFrame()

    data = []
    for run_info in selected_runs:
        row = {
            "Run": (
                run_info["filename"][:20] + "..."
                if len(run_info["filename"]) > 20
                else run_info["filename"]
            ),
            "Date": run_info["date"],
            "Accuracy": f"{run_info['key_metrics']['accuracy']:.4f}",
            "Precision": f"{run_info['key_metrics']['precision']:.4f}",
            "Recall": f"{run_info['key_metrics']['recall']:.4f}",
            "F1": f"{run_info['key_metrics']['f1']:.4f}",
            "ROC-AUC": f"{run_info['key_metrics']['roc_auc']:.4f}",
            "Business Value": f"{run_info['key_metrics'].get('business_value', 0):.0f}",
        }
        data.append(row)

    return pd.DataFrame(data)


def create_param_comparison_df(selected_runs, param_keys=None):
    """Create parameter comparison table."""
    if not selected_runs:
        return pd.DataFrame()

    if not param_keys:
        # Auto-detect common parameters across selected runs
        all_params = set()
        for run_info in selected_runs:  # ✅ Changed from 'run' to 'run_info'
            all_params.update(run_info["key_params"].keys())
        param_keys = sorted(list(all_params))[:8]  # Show top 8 params

    data = []
    for run_info in selected_runs:  # ✅ Changed from 'run' to 'run_info'
        row = {
            "Run": (
                run_info["filename"][:20] + "..."
                if len(run_info["filename"]) > 20
                else run_info["filename"]
            )
        }
        for param in param_keys:
            value = run_info["key_params"].get(param, "N/A")
            if isinstance(value, list):
                value = str(value)[:20] + "..." if len(str(value)) > 20 else str(value)
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
            row[param] = value
        data.append(row)

    return pd.DataFrame(data)


def plot_metrics_comparison(selected_runs):
    """Create interactive metrics comparison chart."""
    if len(selected_runs) < 2:
        return None

    # Prepare data for plotting
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = [
        run_info["filename"][:20] + "..." for run_info in selected_runs
    ]  # ✅ Fixed variable
    fig_data = []

    colors = px.colors.qualitative.Set1

    for i, metric in enumerate(metrics):
        y_values = [
            run_info["key_metrics"][metric] for run_info in selected_runs
        ]  # ✅ Fixed variable
        fig_data.append(
            go.Bar(
                name=metric.replace("_", "-").title(),
                x=x,
                y=y_values,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                text=[f"{y:.3f}" for y in y_values],
                textposition="auto",
            )
        )

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        barmode="group",
        title="Model Metrics Comparison",
        xaxis_title="Model Runs",
        yaxis_title="Score",
        legend_title="Metrics",
        height=500,
        showlegend=True,
    )

    return fig


def main():
    st.set_page_config(
        page_title="Model Comparison Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("📊 Model Evaluation Comparison Dashboard")
    st.markdown("---")

    # Load all runs
    with st.spinner("Loading model runs..."):
        all_runs = load_all_runs()

    if not all_runs:
        st.error("No model runs found in evaluation_results directory!")
        st.info("💡 Run your training pipeline first to generate evaluation results.")
        return

    st.success(f"Loaded {len(all_runs)} model runs")

    # Sidebar: Run selection
    st.sidebar.header("Select Model Runs")

    # Get all filenames for dropdown
    filenames = [run["filename"] for run in all_runs]

    # Base run selection
    st.sidebar.subheader("1. Select Base Run")
    base_run_idx = st.sidebar.selectbox(
        "Choose base model:",
        range(len(filenames)),
        format_func=lambda i: f"{filenames[i]} ({all_runs[i]['date']})",
    )

    # Comparison runs selection (up to 4 more)
    st.sidebar.subheader("2. Select Comparison Runs")
    st.sidebar.info("Select up to 4 additional runs to compare")

    selected_indices = [base_run_idx]
    for i in range(4):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            run_idx = st.selectbox(
                f"Run {i+2}:",
                [""] + [j for j in range(len(filenames)) if j != base_run_idx],
                key=f"comp_{i}",
                index=0,
                format_func=lambda j: (
                    f"{filenames[j]} ({all_runs[j]['date']})" if j else "Select..."
                ),
            )
        with col2:
            st.markdown("**✕**")

        if run_idx and run_idx not in selected_indices:
            selected_indices.append(run_idx)
        elif run_idx in selected_indices:
            st.sidebar.warning("Run already selected!")

    # Limit to max 5 total
    selected_indices = selected_indices[:5]
    selected_runs = [all_runs[i] for i in selected_indices]

    # Main content area
    if len(selected_runs) >= 2:
        # Metrics comparison table
        st.subheader("📈 Metrics Comparison")
        col1, col2 = st.columns([2, 1])

        with col1:
            metrics_df = create_comparison_df(selected_runs)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # with col2:
        #     # Metrics chart
        #     fig = plot_metrics_comparison(selected_runs)
        #     if fig:
        #         st.plotly_chart(fig, use_container_width=True)

        # Parameter comparison
        st.subheader("⚙️ Parameter Comparison")

        # Get common parameters across selected runs
        all_params = set()
        for run in selected_runs:
            all_params.update(run["key_params"].keys())
        param_keys = sorted(list(all_params))[:8]  # Show top 8 params

        params_df = create_param_comparison_df(selected_runs, param_keys)
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Parameter differences
        if len(selected_runs) > 1:
            st.subheader("🔄 Parameter Changes vs Base")
            base_run_info = selected_runs[0]  # ✅ Clear variable name
            base_params = base_run_info["key_params"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Changed Parameters:**")
                changes = []
                for run_info in selected_runs[1:]:  # ✅ Fixed: iterate over runs only
                    run_name = run_info["filename"][:20] + "..."
                    for param in run_info[
                        "key_params"
                    ].keys():  # ✅ Iterate over actual params
                        base_val = base_params.get(param)
                        curr_val = run_info["key_params"].get(param)
                        if (
                            base_val != curr_val
                            and base_val is not None
                            and curr_val is not None
                        ):
                            changes.append(
                                {
                                    "Run": run_name,
                                    "Parameter": param,
                                    "Base": base_val,
                                    "New": curr_val,
                                }
                            )

                if changes:
                    changes_df = pd.DataFrame(changes)
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No parameter changes detected!")

            with col2:
                st.markdown("**Business Impact:**")
                business_values = [
                    run_info["key_metrics"]["business_value"]
                    for run_info in selected_runs
                ]  # ✅ Fixed variable
                fig_biz = px.bar(
                    x=[
                        run_info["filename"][:15] + "..." for run_info in selected_runs
                    ],  # ✅ Fixed variable
                    y=business_values,
                    title="Total Business Value",
                    color=business_values,
                    color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(fig_biz, use_container_width=True)

        # Detailed metrics view
        with st.expander("📋 Detailed Metrics (All Runs)"):
            detailed_data = []
            for run in selected_runs:
                row = {"Run": run["filename"]}
                row.update(run["key_metrics"])
                row.update(run["key_params"])
                detailed_data.append(row)

            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df, use_container_width=True)

    else:
        st.warning("Please select at least 2 runs to compare!")
        st.info("1. Select a base run, then 2. Select comparison runs in the sidebar.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Built with ❤️ for ML experimentation**
        *Load model runs from `evaluation_results/` directory*
        """
    )


if __name__ == "__main__":
    main()
