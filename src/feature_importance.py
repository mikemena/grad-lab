import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, accuracy_score
import mlflow
import os
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def get_feature_importance(model, X, y, feature_names, save_dir, model_type="tree", n_repeats=10):
    """Calculate and log feature importance for any model type."""
    if model_type == "tree":
        return _tree_feature_importance(model, feature_names, save_dir)
    elif model_type == "nn":
        return _nn_permutation_importance(model, X, y, feature_names, save_dir, n_repeats)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _get_base_estimator(model):
    """Extract base estimator from calibrated or wrapped models."""
    if hasattr(model, 'calibrated_classifiers_'):
        return model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'estimator') and model.estimator is not None:
        return model.estimator
    return model


def _tree_feature_importance(model, feature_names, save_dir):
    """
    Get feature importance from tree-based models.

    Uses built-in feature_importances_ which is:
    - Random Forest: Mean Decrease in Impurity (Gini importance)
    - XGBoost: Gain (average loss reduction from splits on this feature)
    """
    base_model = _get_base_estimator(model)

    if not hasattr(base_model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute")
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': base_model.feature_importances_,
        'importance_type': 'impurity_decrease'  # MDI for RF, Gain for XGB
    }).sort_values('importance', ascending=False)

    # Save and log
    _save_and_log_importance(importance_df, save_dir, "tree_feature_importance.csv")

    return importance_df


def _nn_permutation_importance(model, X, y, feature_names, save_dir, n_repeats=10):
    """
    Calculate permutation importance for neural networks.

    How it works:
    1. Get baseline score with original data
    2. For each feature:
       a. Shuffle that feature's values (break relationship with target)
       b. Get new score
       c. Importance = baseline_score - shuffled_score
    3. Repeat n_repeats times and average

    Features that hurt performance most when shuffled are most important.
    """
    logger.info(f"Calculating permutation importance for NN ({n_repeats} repeats)...")

    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.array(X)

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = np.array(y)

    # Ensure y is 1D
    y_np = y_np.ravel()

    # Create sklearn-compatible wrapper for PyTorch model
    wrapped_model = _PyTorchWrapper(model)

    # Calculate permutation importance
    result = permutation_importance(
        wrapped_model,
        X_np,
        y_np,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_log_loss',  # Use log loss for probability models
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'importance_std': result.importances_std,
        'importance_type': 'permutation'
    }).sort_values('importance', ascending=False)

    # Save and log
    _save_and_log_importance(importance_df, save_dir, "nn_feature_importance.csv")

    return importance_df


class _PyTorchWrapper:
    """
    Wrapper to make PyTorch models compatible with sklearn's permutation_importance.

    sklearn expects:
    - predict_proba(X) -> probabilities
    - Classes_ attribute
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.classes_ = np.array([0, 1])  # Binary classification

    def predict_proba(self, X):
        """Return probability predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Return 2D array with [P(class=0), P(class=1)]
        probs = probs.ravel()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def _save_and_log_importance(importance_df, save_dir, filename):
    """Save importance to CSV and log to MLflow."""
    os.makedirs(save_dir, exist_ok=True)
    importance_path = os.path.join(save_dir, filename)
    importance_df.to_csv(importance_path, index=False)

    try:
        mlflow.log_artifact(importance_path)
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    # Log top 10 as metrics
    for i, row in importance_df.head(10).iterrows():
        clean_name = row['feature'].replace(' ', '_').replace('/', '_')[:50]
        try:
            mlflow.log_metric(f"importance_{clean_name}", row['importance'])
        except Exception:
            pass

    logger.info(f"✓ Feature importance saved to {importance_path}")
    logger.info(f"Top 5 features:\n{importance_df.head().to_string()}")


def compare_importance_methods(model, X, y, feature_names, save_dir):
    """
    For tree models, compare built-in importance vs permutation importance.

    This helps identify features that are:
    - High MDI but low permutation: May be overfit or spurious
    - Low MDI but high permutation: May be undervalued by the tree
    """
    base_model = _get_base_estimator(model)

    if not hasattr(base_model, "feature_importances_"):
        logger.warning("Model doesn't support built-in importance")
        return None

    # Get both importance types
    mdi_importance = base_model.feature_importances_

    # Convert data if needed
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.array(X)

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy().ravel()
    else:
        y_np = np.array(y).ravel()

    perm_result = permutation_importance(
        model, X_np, y_np,
        n_repeats=10,
        random_state=42,
        scoring='neg_log_loss',
        n_jobs=-1
    )

    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'mdi_importance': mdi_importance,
        'permutation_importance': perm_result.importances_mean,
        'permutation_std': perm_result.importances_std,
    })

    # Add rank columns
    comparison_df['mdi_rank'] = comparison_df['mdi_importance'].rank(ascending=False)
    comparison_df['perm_rank'] = comparison_df['permutation_importance'].rank(ascending=False)
    comparison_df['rank_diff'] = abs(comparison_df['mdi_rank'] - comparison_df['perm_rank'])

    comparison_df = comparison_df.sort_values('permutation_importance', ascending=False)

    # Save
    comparison_path = os.path.join(save_dir, 'importance_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    try:
        mlflow.log_artifact(comparison_path)
    except Exception:
        pass

    logger.info(f"✓ Importance comparison saved to {comparison_path}")
    logger.info("Features with large rank differences may warrant investigation:")
    high_diff = comparison_df[comparison_df['rank_diff'] > 5]
    if len(high_diff) > 0:
        logger.info(high_diff[['feature', 'mdi_rank', 'perm_rank', 'rank_diff']].to_string())

    return comparison_df
