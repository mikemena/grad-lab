from models.predictor import Predictor, ImprovedPredictor, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import pandas as pd
import numpy as np
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def flatten_dict(d, parent_key='', sep='_'):
    if not isinstance(d, dict):
        logger.warning(f"Expected a dictionary, got {type(d)}. Returning empty dict.")
        return {}

    items = []
    seen_keys = set()
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if new_key in seen_keys:
            logger.warning(f"Duplicate key detected: {new_key}")
        seen_keys.add(new_key)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_numeric_metrics(metrics):
    flat_metrics = flatten_dict(metrics)
    return {k: v for k, v in flat_metrics.items() if isinstance(v, (int, float))}


def instantiate_model(model_config, input_dim):
    """Instantiate a model based on config. For tree models, merges tree_params with
    model-specific params (rf_params, xgb_params).For neural networks, passes relevant
    params to the model constructor.
    """
    model_type = model_config["type"]

    if model_type == "nn_basic":
        nn_params = _extract_nn_params(model_config)
        return Predictor(input_dim=input_dim, **nn_params)

    elif model_type == "nn_improved":
        nn_params = _extract_nn_params(model_config)
        return ImprovedPredictor(input_dim=input_dim, **nn_params)

    elif model_type == "logistic":
        nn_params = _extract_nn_params(model_config)
        return LogisticRegression(input_dim=input_dim, **nn_params)

    elif model_type == "rf":
        rf_params = _extract_tree_params(model_config, "rf")
        logger.info(f"Instantiating RandomForest with params: {rf_params}")
        return RandomForestClassifier(**rf_params)

    elif model_type == "xgb":
        xgb_params = _extract_tree_params(model_config, "xgb")
        logger.info(f"Instantiating XGBoost with params: {xgb_params}")
        return XGBClassifier(**xgb_params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _extract_nn_params(model_config):
    """Extract parameters relevant to neural network models."""
    nn_keys = [
        "hidden_dims", "dropout_rate", "activation",
        "use_batch_norm", "use_residual"
    ]
    return {k: v for k, v in model_config.items() if k in nn_keys}


def _extract_tree_params(model_config, model_type):
    """Extract and merge parameters for tree-based models.
    Merges:
    1. tree_params (shared defaults like random_state, n_jobs)
    2. Model-specific params (rf_params or xgb_params)
    """
    # Start with shared tree params
    params = model_config.get("tree_params", {}).copy()

    # Merge model-specific params
    if model_type == "rf":
        specific_params = model_config.get("rf_params", {})
        # RF-specific valid parameters
        valid_keys = [
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "bootstrap", "oob_score", "class_weight", "criterion",
            "random_state", "n_jobs", "verbose", "warm_start", "max_samples"
        ]
    elif model_type == "xgb":
        specific_params = model_config.get("xgb_params", {})
        # XGB-specific valid parameters
        valid_keys = [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha", "scale_pos_weight",
            "tree_method", "eval_metric", "objective", "booster",
            "random_state", "n_jobs", "verbosity", "enable_categorical",
            "min_child_weight", "gamma"
        ]
    else:
        specific_params = {}
        valid_keys = []

    # Merge specific params (override shared params)
    params.update(specific_params)

    # Filter to only valid keys and remove None values
    filtered_params = {
        k: v for k, v in params.items()
        if k in valid_keys and v is not None
    }

    return filtered_params


def load_dataset(file_path, preprocessor, config):
    """Load dataset from Excel file with preprocessing."""
    df = pd.read_excel(file_path)
    target_column = config["data"]["target_column"]
    logger.debug(f"target_column: {target_column}")

    X = df.drop([target_column], axis=1, errors="ignore").values
    y_raw = df[target_column].values
    logger.debug(f"y_raw unique values: {np.unique(y_raw)}")

    # Encode target if categorical/binary
    if preprocessor.target_type in ["binary", "categorical"]:
        y = preprocessor.target_label_encoder.transform(y_raw)
    else:
        y = y_raw

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    logger.debug(f"Loaded {file_path}: X={X_tensor.shape}, y={y_tensor.shape}")

    return X_tensor, y_tensor, y_raw


def load_dataset_for_trees(file_path, preprocessor, config):
    """ Load dataset for tree models (returns numpy arrays instead of tensors).
        This is a convenience function that avoids the tensor->numpy conversion
        in the training script.
    """
    df = pd.read_excel(file_path)
    target_column = config["data"]["target_column"]

    X = df.drop([target_column], axis=1, errors="ignore").values.astype(np.float32)
    y_raw = df[target_column].values

    # Encode target if categorical/binary
    if preprocessor.target_type in ["binary", "categorical"]:
        y = preprocessor.target_label_encoder.transform(y_raw).astype(np.float32)
    else:
        y = y_raw.astype(np.float32)

    logger.debug(f"Loaded {file_path} for trees: X={X.shape}, y={y.shape}")

    return X, y.ravel(), y_raw
