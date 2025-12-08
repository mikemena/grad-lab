import os
import argparse
import time
import json
import tempfile
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import packaging.version
from sklearn import __version__ as sklearn_version
try:
    from sklearn.calibration import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from ruamel.yaml import YAML
import mlflow
import joblib
import pandas as pd
import numpy as np
from data.data_preprocessor import DataPreprocessor
from evaluate import ModelEvaluator
from utils import (
    instantiate_model,
    load_dataset_for_trees,
    _flatten_dict,
    filter_numeric_metrics
)
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)
yaml = YAML()


class TreeModelTrainer:
    def __init__(self, model, config, mlflow_run=None):
        self.model = model
        self.config = config
        self.mlflow_run = mlflow_run
        self.train_losses = []
        self.val_losses = []
        self._is_fitted = False

    def train(self, X_train, y_train, X_val, y_val, config):
        """Train a tree-based model and compute proxy losses."""
        save_dir = config["preprocessing"]["save_dir"]
        save_path = os.path.join(save_dir, "best_model.pkl")

        # Check if model is already fitted (from hypertuning)
        try:
            check_is_fitted(self.model)
            logger.info("Model already fitted from hypertuning, skipping re-fit")
            self._is_fitted = True
        except NotFittedError:
            logger.info("Fitting tree model...")
            self.model.fit(X_train, y_train)
            self._is_fitted = True

        # Apply calibration if enabled
        if config["tuning"].get("calibrate", False):
            logger.info("Calibrating probabilities with sigmoid (Platt scaling)...")

            sk_ver = packaging.version.parse(sklearn_version)

            if sk_ver >= packaging.version.parse("1.6") and FrozenEstimator is not None:
                # New way: Wrap fitted estimator
                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(self.model),
                    method='sigmoid',
                    cv='prefit'
                )
            else:
                # Old way: direct prefit
                calibrated = CalibratedClassifierCV(
                    estimator=self.model,
                    method='sigmoid',
                    cv='prefit'
                )

            # Fit calibrator on validation set
            calibrated.fit(X_val, y_val)
            self.model = calibrated
            logger.info("✓ Calibration complete")

        # Compute probabilities (using potentially calibrated model)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]

        # Compute proxy losses using log_loss
        train_loss = log_loss(y_train, y_train_proba)
        val_loss = log_loss(y_val, y_val_proba)

        self.train_losses = [train_loss]
        self.val_losses = [val_loss]
        best_val_loss = val_loss

        # Save model with joblib
        joblib.dump(self.model, save_path)
        logger.info(f"Tree model saved to {save_path}")

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=0)
        mlflow.log_metric("val_loss", val_loss, step=0)

        return {
            "best_val_loss": best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def hypertune(self, X_train, y_train, X_val, y_val, config):
        """Perform grid search for tree-based models."""
        if not self.config["tuning"].get("enabled", False):
            logger.info("Tuning disabled in config; skipping.")
            return None

        logger.info("Starting grid search for tree-based model")
        model_type = self.config["model"]["type"]

        # Define parameter grid based on model type
        param_grid = {}
        if model_type == "rf":
            param_grid = {
                "n_estimators": self.config["tuning"].get("n_estimators_range", [100, 200, 400]),
                "max_depth": self.config["tuning"].get("max_depth_range", [5, 10, 20, None]),
                "min_samples_split": self.config["tuning"].get("min_samples_split_range", [2, 5, 10]),
                "min_samples_leaf": self.config["tuning"].get("min_samples_leaf_range", [1, 2, 4]),
            }
        elif model_type == "xgb":
            param_grid = {
                "n_estimators": self.config["tuning"].get("n_estimators_range", [100, 200, 400]),
                "max_depth": self.config["tuning"].get("max_depth_range", [3, 6, 9]),
                "learning_rate": self.config["tuning"].get("lr_range", [0.01, 0.05, 0.1]),
                "subsample": self.config["tuning"].get("subsample_range", [0.6, 0.8, 1.0]),
            }
        else:
            logger.warning(f"No tuning grid defined for model type: {model_type}")
            return None

        if not param_grid:
            logger.warning("Empty parameter grid; skipping tuning.")
            return None

        with mlflow.start_run(nested=True, run_name=f"grid_search_{model_type}"):
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring="neg_log_loss",
                cv=3,
                n_jobs=self.config["model"].get("tree_params", {}).get("n_jobs", -1),
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            # Log best params and score
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert to positive log_loss
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_cv_log_loss", best_score)

            # Log full grid search results
            cv_results = grid_search.cv_results_
            for i, params in enumerate(cv_results["params"]):
                mean_score = -cv_results["mean_test_score"][i]
                mlflow.log_metric("cv_log_loss", mean_score, step=i)

            # Update model with best estimator (already fitted!)
            self.model = grid_search.best_estimator_
            self._is_fitted = True

            # Update config with best params for logging
            self.config["model"].update(best_params)

            logger.info(f"✓ Best params: {best_params}")
            logger.info(f"✓ Best CV log_loss: {best_score:.4f}")

        return best_params


def get_base_estimator(model):
    """
    Extract the base estimator from a potentially wrapped model.
    Handles CalibratedClassifierCV and other wrappers.
    """
    if hasattr(model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV stores calibrated classifiers
        return model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'estimator') and model.estimator is not None:
        return model.estimator
    return model


def log_feature_importance(model, feature_names, save_dir):
    """Log feature importance for tree-based models."""
    base_model = get_base_estimator(model)

    if hasattr(base_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save to CSV
        importance_path = os.path.join(save_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Log top 10 as metrics
        for i, row in importance_df.head(10).iterrows():
            # Clean feature name for MLflow metric naming
            clean_name = row['feature'].replace(' ', '_').replace('/', '_')[:50]
            mlflow.log_metric(f"importance_{clean_name}", row['importance'])

        logger.info(f"✓ Feature importance saved to {importance_path}")
        logger.info(f"Top 5 features:\n{importance_df.head().to_string()}")

        return importance_df
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return None


def log_tree_params(model, config):
    """Log final tree model parameters to MLflow."""
    base_model = get_base_estimator(model)

    if hasattr(base_model, "get_params"):
        final_params = base_model.get_params()

        # Filter only tree-relevant params
        tree_keys = [
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "learning_rate", "subsample", "colsample_bytree", "reg_lambda",
            "criterion", "class_weight", "scale_pos_weight", "max_features"
        ]
        tree_params = {
            k: v for k, v in final_params.items()
            if k in tree_keys and v is not None
        }

        # Log with prefix
        mlflow.log_params({f"final_{k}": v for k, v in tree_params.items()})
        logger.info(f"Logged final tree params: {tree_params}")
        return tree_params
    else:
        logger.warning("Model does not have get_params(); skipping param logging.")
        return None


def main(config_path):
    start_time = time.time()
    with open(config_path, "r") as f:
        config = yaml.load(f)

    experiment_name = config['training'].get('experiment_name')
    logger.debug(f"Experiment Name: {experiment_name}")

    mlflow.set_experiment(f"{experiment_name}")
    flat_config = {
        k: v for k, v in _flatten_dict(config).items()
        if isinstance(v, (str, int, float, bool))
    }

    with mlflow.start_run(run_name=config["training"].get("run_name", "default")):
        mlflow.log_params(flat_config)
        mlflow.log_artifact(config_path)

        # Log model type
        model_type = config["model"]["type"]
        mlflow.set_tag("model_type", model_type)
        logger.info(f"Training {model_type.upper()} model")

        # Extract save_dir
        save_dir = config.get("preprocessing", {}).get(
            "save_dir", "experiments/preprocessing/artifacts"
        )
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Using save_dir: {save_dir}")

        # Load data file paths
        state_file = config["data"]["filepath"]["state"]
        train_file = config["data"]["filepath"]["train"]
        val_file = config["data"]["filepath"]["val"]
        test_file = config["data"]["filepath"]["test"]

        # Initialize preprocessor and load state
        preprocessor = DataPreprocessor(save_dir=save_dir)
        preprocessor.load_state(state_file)

        # Load datasets as numpy arrays (using the new utility function)
        X_train, y_train, y_train_raw = load_dataset_for_trees(train_file, preprocessor, config)
        X_val, y_val, _ = load_dataset_for_trees(val_file, preprocessor, config)
        X_test, y_test, _ = load_dataset_for_trees(test_file, preprocessor, config)

        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Validate input dimensions against config
        config_input_dim = config['model'].get('input_dim')
        if config_input_dim not in [None, 'auto'] and X_train.shape[1] != config_input_dim:
            logger.warning(
                f"Dataset has {X_train.shape[1]} features, "
                f"but config expects {config_input_dim}"
            )

        # Load feature names from preprocessor state
        with open(state_file, "r") as f:
            preprocessor_state = json.load(f)
            feature_names = preprocessor_state["feature_columns"]
        logger.info(f"Loaded {len(feature_names)} feature names")

        # Instantiate model with config parameters
        model = instantiate_model(config["model"], input_dim)
        logger.info(f"Model instantiated: {type(model).__name__}")

        # Create trainer
        trainer = TreeModelTrainer(model, config)

        # Hypertune (if enabled, model will be fitted during grid search)
        best_params = trainer.hypertune(X_train, y_train, X_val, y_val, config)

        # Train (will skip fit if already fitted from hypertuning)
        training_results = trainer.train(X_train, y_train, X_val, y_val, config)

        # Get the trained model (potentially updated by trainer)
        model = trainer.model

        # Log final tree parameters
        final_params = log_tree_params(model, config)

        # Log feature importance
        importance_df = log_feature_importance(model, feature_names, save_dir)

        # Log scalar metrics
        mlflow.log_metric("best_val_loss", training_results["best_val_loss"])

        # Save losses as JSON artifact
        losses_data = {
            "train_losses": training_results["train_losses"],
            "val_losses": training_results["val_losses"]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(losses_data, f)
            losses_path = f.name
        mlflow.log_artifact(losses_path)
        os.unlink(losses_path)

        # === EVALUATION ===
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATING BEST MODEL")
        logger.info("=" * 50)

        evaluator = ModelEvaluator(
            model=model,
            device=None,  # Trees don't use device
            save_dir="evaluation_results",
            config_path=config_path,
        )
        metrics, predictions, probabilities, targets = evaluator.evaluate(
            [(X_test, y_test)],  # Pass as list for compatibility
            feature_names=feature_names
        )

        # Log evaluation metrics
        serial_metrics = evaluator._convert_to_serializable(metrics)
        flat_metrics = filter_numeric_metrics(serial_metrics)
        mlflow.log_metrics(flat_metrics)

        # === SAVE RESULTS ===
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M")
        filename = f"final_results_{model_type}_{timestamp}.json"

        results = {
            "date": datetime.now().strftime("%m-%d-%Y %H:%M"),
            "model_type": model_type,
            "config_path": config_path,
            "data_files": {
                "train": train_file,
                "val": val_file,
                "test": test_file
            },
            "input_dim": input_dim,
            "best_hyperparams": best_params,
            "final_model_params": final_params,
            "training_results": {
                "best_val_loss": training_results["best_val_loss"],
                "train_loss": training_results["train_losses"][-1] if training_results["train_losses"] else None,
                "val_loss": training_results["val_losses"][-1] if training_results["val_losses"] else None,
            },
            "test_metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics.get("log_loss"),
                "average_precision": metrics.get("average_precision"),
            },
            "confusion_matrix": {
                "true_positives": metrics.get("true_positives"),
                "true_negatives": metrics.get("true_negatives"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
            },
            "full_config": config,
        }

        # Save results JSON
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(filepath)
        logger.info(f"Results saved to {filepath}")

        # === LOG MODEL TO MLFLOW ===
        base_model = get_base_estimator(model)

        if isinstance(base_model, RandomForestClassifier):
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="PredictorModel"
            )
            logger.info("Model logged to MLflow as sklearn model")
        elif isinstance(base_model, XGBClassifier):
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name="PredictorModel"
            )
            logger.info("Model logged to MLflow as xgboost model")
        else:
            # Fallback: save with joblib and log as artifact
            model_path = os.path.join(save_dir, "final_model.pkl")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            logger.info(f"Model saved as artifact: {model_path}")

        # Log runtime inside the run
        end_time = time.time()
        total_runtime = (end_time - start_time) / 60
        mlflow.log_metric("total_runtime_minutes", total_runtime)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("🎉 TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info(f"Model type: {model_type.upper()}")
    logger.info(f"Total runtime: {total_runtime:.2f} minutes")
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    if best_params:
        logger.info(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tree-based models (Random Forest, XGBoost)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
