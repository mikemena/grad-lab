import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from models.predictor import Predictor, ImprovedPredictor, FocalLoss  # Relative import
from logger import setup_logger
from data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
)
import pandas as pd
from datetime import datetime
import json
from evaluate import ModelEvaluator
from sklearn.utils.class_weight import compute_class_weight

logger = setup_logger(__name__, include_location=True)


class ModelTrainer:
    """Unified trainer (consolidate from v1-v4)."""

    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        # Handle loss based on config
        loss_type = config["training"]["loss_type"]
        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=config["training"]["alpha"], gamma=config["training"]["gamma"]
            )
        elif loss_type == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        # Class weights if enabled
        self.class_weights = None
        if config["training"]["use_class_weights"]:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train_raw), y=y_train_raw
            )
            # MUCH more conservative weighting
            class_weights[1] *= 1.5  # Only 1.5x boost, not 3x
            class_weights = np.clip(
                class_weights, 0.1, 10
            )  # Much more reasonable range
            logger.info(f"Conservative class weights: {class_weights}")

    # train_epoch(), validate(), train() methods adapted from your v4 (with early stopping, scheduler, etc.)
    # evaluate() from your code...

    def hypertune(self, train_loader, val_loader):
        """Run simple grid search if tuning enabled."""
        if not self.config["tuning"]["enabled"]:
            return
        best_val_loss = float("inf")
        best_params = {}
        for lr in self.config["tuning"]["lr_range"]:
            for hidden_dims in self.config["tuning"]["hidden_dims_options"]:
                for dropout in self.config["tuning"]["dropout_range"]:
                    logger.info(
                        f"Tuning: lr={lr}, hidden_dims={hidden_dims}, dropout={dropout}"
                    )
                    # Re-instantiate model with these params
                    temp_config = self.config.copy()
                    temp_config["model"]["hidden_dims"] = hidden_dims
                    temp_config["model"]["dropout_rate"] = dropout
                    temp_model = instantiate_model(
                        temp_config["model"], input_dim
                    )  # Define below
                    temp_trainer = ModelTrainer(temp_model, temp_config)
                    results = temp_trainer.train(
                        train_loader, val_loader, lr=lr
                    )  # Pass overrides
                    if results["best_val_loss"] < best_val_loss:
                        best_val_loss = results["best_val_loss"]
                        best_params = {
                            "lr": lr,
                            "hidden_dims": hidden_dims,
                            "dropout": dropout,
                        }
        logger.info(f"Best tuned params: {best_params}")
        # Save best params to YAML or use for final training


def instantiate_model(model_config, input_dim):
    model_type = model_config["type"]
    if model_type == "basic":
        return Predictor(input_dim=input_dim, **model_config)
    elif model_type == "improved":
        return ImprovedPredictor(input_dim=input_dim, **model_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data (adapt from your load_dataset/create_data_loaders)
    preprocessor = DataPreprocessor()
    # ... load state, data from config['data']['filepath']
    X_train, y_train = ...  # Your code
    input_dim = X_train.shape[1]

    # Compute class weights if needed
    class_weights = (
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        if config["training"]["use_class_weights"]
        else None
    )

    # Create loaders (with sampler if imbalance)
    train_loader, val_loader, test_loader = create_data_loaders(...)  # Your function

    # Instantiate model
    model = instantiate_model(config["model"], input_dim)

    # Trainer
    trainer = ModelTrainer(model, config)
    trainer.hypertune(train_loader, val_loader)  # If enabled

    # Train
    training_results = trainer.train(train_loader, val_loader, **config["training"])

    logger.info("\n=== EVALUATING BEST MODEL ===")
    evaluator = ModelEvaluator(
        model=best_model, device=best_trainer.device, save_dir="evaluation_results"
    )

    # Set the optimal threshold
    evaluator.threshold = best_trainer.best_threshold

    metrics, predictions, probabilities, targets = evaluator.evaluate(
        test_loader, feature_names=feature_names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
