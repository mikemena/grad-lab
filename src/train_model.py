import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger
from data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime
import json
from visualize import ModelVisualizer

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):
    """Simple Feedforward Neural Network for predicting classification target"""

    def __init__(self, input_dim, hidden_dims=[64], dropout_rate=0.0):
        super(Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network layers (simplified: no batch norm for starting point)
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class ModelTrainer:
    """Handles model training, validation, and evaluation"""

    def __init__(self, model, device=None, class_weights=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            if self.class_weights is not None:
                # Apply per-sample weights
                weight = self.class_weights[batch_y.long()]
                loss = criterion(outputs, batch_y)
                loss = (loss * weight).mean()  # Apply weights and compute mean loss
            else:
                loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                if self.class_weights is not None:
                    # Apply per-sample weights
                    weight = self.class_weights[batch_y.long()]
                    loss = criterion(outputs, batch_y)
                    loss = (loss * weight).mean()  # Apply weights and compute mean loss
                else:
                    loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0,
        patience=10,
        min_delta=1e-4,
        save_path="best_model.pt",
        optimizer_name="Adam",
    ):
        """Train the model with early stopping (simplified: no scheduler, no clipping)"""
        # binary classification
        # Set reduction='none' for class weights to allow per-sample weighting
        criterion = nn.BCEWithLogitsLoss(
            reduction="none" if self.class_weights is not None else "mean"
        )

        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training on {self.device}")
        logger.info(f"Model Info: {self.model.get_model_info()}")

        if self.class_weights is not None:
            logger.info(f"Using class weights: {self.class_weights.tolist()}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_config": self.model.get_model_info(),
                        "class_weights": (
                            self.class_weights.tolist()
                            if self.class_weights is not None
                            else None
                        ),
                    },
                    save_path,
                )
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                probabilities = torch.sigmoid(logits)
                predictions.extend(probabilities.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        # Apply threshold to get binary predictions
        binary_preds = (predictions >= 0.5).astype(int)

        # Classification metrics
        acc = accuracy_score(targets, binary_preds)
        f1 = f1_score(targets, binary_preds)
        try:
            auc = roc_auc_score(targets, predictions)  # Use raw scores for AUC
        except ValueError:
            auc = None  # Handle case where only one class is present in y_true

        cm = confusion_matrix(targets, binary_preds)

        # Percentage of negatives caught - True Negative Rate
        tn, fp, fn, tp = cm.ravel()  # Extract TN, FP, FN, TP
        tnr = tn / (tn + fp)

        # Percentage of positives caught - True Positive Rate
        recall = recall_score(targets, binary_preds)

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "recall": recall,
            "tnr": tnr,
        }

        logger.info("Test Set Evaluation:")
        logger.info(f"TNR: {tnr:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"f1 Score: {f1:.4f}")
        logger.info(f"roc auc: {auc:.4f}")
        logger.info(f"Confusion Matrix: {cm}")

        return metrics, predictions, targets


def load_dataset(file_path):
    # Load the preprocessor state
    preprocessor = DataPreprocessor()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    state_file = os.path.join(
        root_dir, "preprocessing_artifacts", "preprocessor_state.json"
    )
    preprocessor.load_state(state_file)

    df = pd.read_excel(file_path)
    X = df.drop(["personal_loan", "temp_index"], axis=1, errors="ignore").values
    y = df["personal_loan"].values
    y_raw = y.copy()  # Preserve raw y for class weights

    # Encode y if binary or categorical target
    if preprocessor.target_type in ["binary", "categorical"]:
        if preprocessor.target_label_encoder is None:
            logger.error(
                "Target LabelEncoder not loaded. Ensure preprocessor state is saved."
            )
            raise ValueError(
                "Target LabelEncoder not loaded. Ensure preprocessor state is saved."
            )
        y = preprocessor.target_label_encoder.transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, y_raw


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create DataLoaders for training, validation, and testing"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train model with config")
    parser.add_argument("--config", type=str, default="configs/model_v1.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Make paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_artifacts_dir = os.path.join(script_dir, "../preprocessing_artifacts")

    # Debug prints to verify paths
    logger.info("\nDebug Info:")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Preprocessing artifacts directory: {preprocessing_artifacts_dir}")

    state_file = os.path.join(preprocessing_artifacts_dir, "preprocessor_state.json")
    train_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_train_processed.xlsx"
    )
    val_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_val_processed.xlsx"
    )
    test_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_test_processed.xlsx"
    )

    # Check each file explicitly
    files_to_check = [state_file, train_file, val_file, test_file]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        logger.error(
            "Run the preprocessing step (e.g., prepare.py or similar) to generate them, or adjust the paths if incorrect."
        )
        return None

    # Load preprocessor state to get feature names for plotting
    with open(state_file, "r") as f:
        preprocessor_state = json.load(f)
    feature_names = preprocessor_state["feature_columns"]

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load processed data from Excel
    logger.info("\n=== LOADING PROCESSED DATA FROM EXCEL ===")
    X_train, y_train, y_train_raw = load_dataset(train_file)
    X_val, y_val, _ = load_dataset(val_file)
    X_test, y_test, _ = load_dataset(test_file)

    input_dim = X_train.shape[1]

    class_weights = None
    if config['training'].get('use_class_weights', False):
        logger.info("\nComputing class weights...")
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_raw), y=y_train_raw
        )
        logger.info(f"Class weights: {class_weights}")

    model = Predictor(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate']
    )
    trainer = ModelTrainer(model, class_weights=class_weights)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=config['data']['batch_size']
    )
    training_results = trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['epochs'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        save_path="models/best_model.pt",
        optimizer_name=config['training']['optimizer_name'],
    )
    metrics, predictions, targets = trainer.evaluate(test_loader)

    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir="plots")

    # Generate visualizations
    visualizer.plot_training_history(
        trainer.train_losses, trainer.val_losses, display=False, save=True
    )
    visualizer.plot_confusion_matrix(targets, predictions, display=False, save=True)
    visualizer.plot_data_distribution(
        X_train.numpy(), feature_names, display=False, save=True
    )
    visualizer.plot_prediction_distribution(
        predictions, targets, display=False, save=True
    )
    visualizer.plot_roc_curve(targets, predictions, display=False, save=True)
    visualizer.plot_precision_recall_curve(
        targets, predictions, display=False, save=True
    )
    visualizer.plot_metrics_bar(metrics, display=False, save=True)

    # Save final results
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_config": model.get_model_info(),
        "training_results": training_results,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "recall": metrics["recall"],
            "tnr": metrics["tnr"],
        },
        "preprocessing_artifacts": state_file,
        "config_used": config,
    }

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n🎉 TRAINING COMPLETED!")

    return model, trainer, results


if __name__ == "__main__":
    result = main()
    if result:
        logger.info("\n✨ Pipeline completed!")
    else:
        logger.info("\n❌ Failed. Fix issues and retry.")