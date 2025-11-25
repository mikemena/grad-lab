from models.predictor import Predictor, ImprovedPredictor, LogisticRegression  # Relative import
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import pandas as pd
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
    model_type = model_config["type"]
    # Create a copy of model_config without input_dim and type
    model_params = model_config.copy()
    if "input_dim" in model_params:
        del model_params["input_dim"]
    if "type" in model_params:
        del model_params["type"]

    if model_type == "nn_basic":
        return Predictor(input_dim=input_dim, **model_params)
    elif model_type == "nn_improved":
        return ImprovedPredictor(input_dim=input_dim, **model_params)
    elif model_type == "logistic":
        return LogisticRegression(input_dim=input_dim, **model_params)
    elif model_type == "rf":
        return RandomForestClassifier()
    elif model_type == 'xgb':
        return XGBClassifier()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def load_dataset(file_path, preprocessor, config):
    """Load dataset with preprocessing."""
    # preprocessor = DataPreprocessor(save_dir=save_dir)
    # logger.debug(f"Loading state for: {file_path}")
    # preprocessor.load_state(state_file)
    df = pd.read_excel(file_path)
    target_column = config["data"]["target_column"]
    logger.debug(f"trget_colun: {target_column}")
    X = df.drop([target_column], axis=1, errors="ignore").values
    y_raw = df[target_column].values
    logger.debug(f"y_raw: {y_raw}")
    if preprocessor.target_type in ["binary", "categorical"]:
        y = preprocessor.target_label_encoder.transform(y_raw)
    else:
        y = y_raw
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, y_raw
