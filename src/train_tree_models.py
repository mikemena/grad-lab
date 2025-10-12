import os
import argparse
import time
import json
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from ruamel.yaml import YAML
import mlflow
import joblib
from train_model import load_dataset, instantiate_model, _flatten_dict, DataPreprocessor
from evaluate import ModelEvaluator
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

    def train(self, X_train, y_train, X_val, y_val, config):
        """Train a tree-based model and compute proxy losses."""
        save_dir = config["preprocessing"]["save_dir"]
        save_path = os.path.join(save_dir, "best_model.pkl")

        # Fit the model (class weights handled via config params in instantiate_model)
        self.model.fit(X_train, y_train)

        # Compute proxy losses using log_loss (for consistency with PyTorch metrics)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
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
        """Perform grid search for tree-based models (placeholder)."""
        if not self.config["tuning"]["enabled"]:
            logger.info("Tuning disabled in config; skipping.")
            return

        logger.info("Starting grid search for tree-based model")
        model_type = self.config["model"]["type"]

        # Define parameter grid based on model type
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
            return

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
            best_score = -grid_search.best_score_  # Convert back to positive log_loss
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_loss", best_score)
            logger.info(f"Best params: {best_params}, Best val loss: {best_score:.4f}")

            # Update model with best params
            self.model = grid_search.best_estimator_
            self.config["model"].update(best_params)  # Update config for consistency

def main(config_path):
    start_time = time.time()
    with open(config_path, "r") as f:
        config = yaml.load(f)

    mlflow.set_experiment("MyProject")
    flat_config = {k: v for k, v in _flatten_dict(config).items() if isinstance(v, (str, int, float, bool))}

    with mlflow.start_run(run_name=config["training"].get("run_name", "default")):
        mlflow.log_params(flat_config)
        mlflow.log_artifact(config_path)

        # Extract save_dir
        save_dir = config.get("preprocessing", {}).get("save_dir", "experiments/preprocessing/artifacts")
        logger.info(f"Using save_dir: {save_dir}")

        # Load data
        state_file = config["data"]["filepath"]["state"]
        train_file = config["data"]["filepath"]["train"]
        val_file = config["data"]["filepath"]["val"]
        test_file = config["data"]["filepath"]["test"]

        preprocessor = DataPreprocessor(save_dir=save_dir)
        preprocessor.load_state(state_file)

        X_train, y_train, y_train_raw = load_dataset(train_file, preprocessor, config)
        X_val, y_val, _ = load_dataset(val_file, preprocessor, config)
        X_test, y_test, _ = load_dataset(test_file, preprocessor, config)

        input_dim = X_train.shape[1]
        logger.debug(f"input_dim: {input_dim}")
        if X_train.shape[1] != config['model']['input_dim']:
            logger.warning(f"Dataset has {X_train.shape[1]} features, but config expects {config['model']['input_dim']}")

        with open(state_file, "r") as f:
            preprocessor_state = json.load(f)
            feature_names = preprocessor_state["feature_columns"]

        # Convert tensors to NumPy for tree models
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy().ravel()
        X_val_np = X_val.cpu().numpy()
        y_val_np = y_val.cpu().numpy().ravel()
        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy().ravel()

        # Instantiate model
        model = instantiate_model(config["model"], input_dim)

        # Trainer
        trainer = TreeModelTrainer(model, config)
        trainer.hypertune(X_train_np, y_train_np, X_val_np, y_val_np, config)

        # Train
        training_results = trainer.train(X_train_np, y_train_np, X_val_np, y_val_np, config)

        # Log scalar metrics
        mlflow.log_metric("best_val_loss", training_results["best_val_loss"])

        # Save losses as JSON artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"train_losses": training_results["train_losses"], "val_losses": training_results["val_losses"]}, f)
        mlflow.log_artifact(f.name)
        os.unlink(f.name)

        logger.info("\n=== EVALUATING BEST MODEL ===")
        evaluator = ModelEvaluator(
            model=model,
            device=None,  # Trees don't use device
            save_dir="evaluation_results",
            config_path=config_path,
        )
        metrics, predictions, probabilities, targets = evaluator.evaluate(
            [(X_test_np, y_test_np)],  # Pass as list for compatibility with evaluate.py
            feature_names=feature_names
        )

        # Log metrics
        serial_metrics = evaluator._convert_to_serializable(metrics)
        flat_metrics = {k: v for k, v in serial_metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(flat_metrics)

        # Save results
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M")
        filename = f"final_results_{timestamp}.json"
        results = {
            "date": datetime.now().strftime("%m-%d-%Y %H:%M"),
            "full_config": config,
            "model_config": config["model"],
            "training_results": training_results,
            "test_metrics": {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "recall": metrics["recall"],
            },
        }
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        # Log model
        if isinstance(model, RandomForestClassifier):
            mlflow.sklearn.log_model(model, "model", registered_model_name="PredictorModel")
        elif isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, "model", registered_model_name="PredictorModel")
        else:
            logger.warning("Unknown model type; skipping MLflow model logging.")

    end_time = time.time()
    total_runtime = (end_time - start_time) / 60
    mlflow.log_metric("total_runtime_minutes", total_runtime)
    logger.info(f"Training Completed: Total runtime: {total_runtime:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
