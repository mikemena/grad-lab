import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from models.predictor import Predictor, ImprovedPredictor, FocalLoss  # Relative import
from logger import setup_logger
from data.data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from datetime import datetime
import json
from evaluate import ModelEvaluator
from sklearn.utils.class_weight import compute_class_weight

logger = setup_logger(__name__, include_location=True)


class ModelTrainer:
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
        # Class weights if enabled (to be set later with actual data)
        self.class_weights = None

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            if self.class_weights is not None:
                weight = self.class_weights[batch_y.long()]
                loss = (loss * weight).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                if self.class_weights is not None:
                    weight = self.class_weights[batch_y.long()]
                    loss = (loss * weight).mean()
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def train(self, train_loader, val_loader, config, **kwargs):
        """Train the model with early stopping."""
        epochs = kwargs.get("epochs", 50)
        lr = kwargs.get("lr", 0.001)
        weight_decay = kwargs.get("weight_decay", 0.0)
        patience = kwargs.get("patience", 10)
        min_delta = kwargs.get("min_delta", 1e-4)
        optimizer_name = kwargs.get("optimizer_name", "Adam")
        save_dir = config["preprocessing"]["save_dir"]
        save_path = os.path.join(save_dir, "best_model.pt")

        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, self.criterion, optimizer)
            val_loss = self.validate(val_loader, self.criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save({"model_state_dict": self.model.state_dict()}, save_path)
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load(save_path)["model_state_dict"])
        return {
            "best_val_loss": best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def hypertune(self, train_loader, val_loader, input_dim):
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
                    temp_config = self.config.copy()
                    temp_config["model"]["hidden_dims"] = hidden_dims
                    temp_config["model"]["dropout_rate"] = dropout
                    temp_model = instantiate_model(temp_config["model"], input_dim)
                    temp_trainer = ModelTrainer(temp_model, temp_config)
                    results = temp_trainer.train(train_loader, val_loader, lr=lr)
                    if results["best_val_loss"] < best_val_loss:
                        best_val_loss = results["best_val_loss"]
                        best_params = {
                            "lr": lr,
                            "hidden_dims": hidden_dims,
                            "dropout": dropout,
                        }
        logger.info(f"Best tuned params: {best_params}")
        # Apply best params (optional)
        if best_params:
            self.config["training"]["lr"] = best_params["lr"]
            self.config["model"]["hidden_dims"] = best_params["hidden_dims"]
            self.config["model"]["dropout_rate"] = best_params["dropout"]
            self.model = instantiate_model(self.config["model"], input_dim)


def instantiate_model(model_config, input_dim):
    model_type = model_config["type"]
    # Create a copy of model_config without input_dim and type
    model_params = model_config.copy()
    if "input_dim" in model_params:
        del model_params["input_dim"]
    if "type" in model_params:
        del model_params["type"]

    if model_type == "basic":
        return Predictor(input_dim=input_dim, **model_params)
    elif model_type == "improved":
        return ImprovedPredictor(input_dim=input_dim, **model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_dataset(file_path, state_file, save_dir, config):
    """Load dataset with preprocessing."""
    preprocessor = DataPreprocessor(save_dir=save_dir)
    preprocessor.load_state(state_file)
    df = pd.read_excel(file_path)
    target_column = config["data"]["target_column"]
    X = df.drop([target_column, "temp_index"], axis=1, errors="ignore").values
    y_raw = df[target_column].values
    if preprocessor.target_type in ["binary", "categorical"]:
        y = preprocessor.target_label_encoder.transform(y_raw)
    else:
        y = y_raw
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, y_raw


def create_data_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, use_sampler=False
):
    """Create DataLoaders with optional weighted sampling."""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    if use_sampler:
        sample_weights = np.ones(len(y_train))  # Placeholder, adjust based on imbalance
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract save_dir from config
    save_dir = config.get("preprocessing", {}).get(
        "save_dir", "experiments/preprocessing/artifacts"
    )
    logger.info(f"Using save_dir: {save_dir}")

    # Load data
    # preprocessor = DataPreprocessor()
    state_file = config["data"]["filepath"]["state"]
    train_file = config["data"]["filepath"]["train"]
    val_file = config["data"]["filepath"]["val"]
    test_file = config["data"]["filepath"]["test"]

    X_train, y_train, y_train_raw = load_dataset(
        train_file, state_file, save_dir, config
    )
    X_val, y_val, _ = load_dataset(val_file, state_file, save_dir, config)
    X_test, y_test, _ = load_dataset(test_file, state_file, save_dir, config)

    input_dim = X_train.shape[1]
    with open(state_file, "r") as f:
        preprocessor_state = json.load(f)
    feature_names = preprocessor_state["feature_columns"]

    # Compute class weights if needed
    if config["training"]["use_class_weights"]:
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_raw), y=y_train_raw
        )
        class_weights[1] *= 1.5  # Conservative boost
        class_weights = np.clip(class_weights, 0.1, 10)
        logger.info(f"Conservative class weights: {class_weights}")
    else:
        class_weights = None

    # Create loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=config["training"]["batch_size"],
        use_sampler=config["training"].get("use_sampler", False),
    )

    # Instantiate model
    model = instantiate_model(config["model"], input_dim)

    # Trainer
    trainer = ModelTrainer(model, config)
    trainer.hypertune(train_loader, val_loader, input_dim)  # Pass input_dim

    # Train
    training_results = trainer.train(
        train_loader, val_loader, config, **config["training"]
    )

    logger.info("\n=== EVALUATING BEST MODEL ===")
    evaluator = ModelEvaluator(
        model=model, device=trainer.device, save_dir="evaluation_results"
    )
    metrics, predictions, probabilities, targets = evaluator.evaluate(
        test_loader, feature_names=feature_names
    )

    # Save results
    results = {
        "date": datetime.now().strftime("%m-%d-%Y %H:%M"),
        "model_config": config["model"],
        "training_results": training_results,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
        },
    }
    # final_json_file = r"{save_dir}/final_results.json"
    # logger.debug(f"")
    with open(f"{save_dir}/final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n🎉 FINAL TRAINING COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
