import os
import argparse
from ruamel.yaml import YAML
import torch
import torch.nn as nn
import torch.optim as optim
from logger import setup_logger
from data.data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from models.predictor import FocalLoss
from utils import instantiate_model, load_dataset, _flatten_dict, filter_numeric_metrics
# from feature_importance import get_feature_importance
import numpy as np
from datetime import datetime
import json
from evaluate import ModelEvaluator
from sklearn.utils.class_weight import compute_class_weight
import time
import tempfile
import mlflow

logger = setup_logger(__name__, include_location=True)
yaml = YAML()

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

        # Learning rate scheduler
        scheduler = None
        if kwargs.get("use_scheduler", False):
            scheduler_type = kwargs.get("scheduler_type", "cosine")
            if scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            elif scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            elif scheduler_type == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
            logger.info(f"Using {scheduler_type} learning rate scheduler")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, self.criterion, optimizer)
            val_loss = self.validate(val_loader, self.criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save({"model_state_dict": self.model.state_dict()}, save_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
                )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(save_path)["model_state_dict"])
        logger.info(f"Loaded best model from {save_path}")

        return {
            "best_val_loss": best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epochs_trained": len(self.train_losses),
        }

    def hypertune(self, train_loader, val_loader, input_dim, config):
        """Hyperparameter tuning via grid search."""
        if not self.config["tuning"].get("enabled", False):
            logger.info("Tuning disabled in config; skipping.")
            return None

        logger.info("Starting hyperparameter tuning...")
        best_val_loss = float("inf")
        best_params = {}

        for lr in self.config["tuning"]["lr_range"]:
            with mlflow.start_run(nested=True, run_name=f"trial_lr_{lr}"):
                mlflow.log_param("lr", lr)

                if self.config["model"]["type"] == "logistic":
                    logger.info(f"Tuning: lr={lr} (logistic regression)")
                    temp_config = self.config.copy()
                    temp_model = instantiate_model(temp_config["model"], input_dim)
                    temp_trainer = ModelTrainer(temp_model, temp_config, self.device)
                    results = temp_trainer.train(
                        train_loader, val_loader, config, lr=lr,
                        epochs=self.config["training"]["epochs"],
                        patience=self.config["training"]["patience"]
                    )
                    mlflow.log_metric("val_loss", results["best_val_loss"])

                    if results["best_val_loss"] < best_val_loss:
                        best_val_loss = results["best_val_loss"]
                        best_params = {"lr": lr}
                else:
                    for hidden_dims in self.config["tuning"]["hidden_dims_options"]:
                        for dropout in self.config["tuning"]["dropout_range"]:
                            logger.info(
                                f"Tuning: lr={lr}, hidden_dims={hidden_dims}, dropout={dropout}"
                            )
                            temp_config = self.config.copy()
                            temp_config["model"]["hidden_dims"] = hidden_dims
                            temp_config["model"]["dropout_rate"] = dropout
                            temp_model = instantiate_model(temp_config["model"], input_dim)
                            temp_trainer = ModelTrainer(temp_model, temp_config, self.device)
                            results = temp_trainer.train(
                                train_loader, val_loader, config, lr=lr,
                                epochs=self.config["training"]["epochs"],
                                patience=self.config["training"]["patience"]
                            )
                            mlflow.log_metric("val_loss", results["best_val_loss"])

                            if results["best_val_loss"] < best_val_loss:
                                best_val_loss = results["best_val_loss"]
                                best_params = {
                                    "lr": lr,
                                    "hidden_dims": hidden_dims,
                                    "dropout": dropout,
                                }
                                logger.info(f"New best params: {best_params}")

        if best_params:
            logger.info(f"✓ Best tuned params: {best_params}")
            self.config["training"]["lr"] = best_params["lr"]
            if self.config["model"]["type"] != "logistic":
                self.config["model"]["hidden_dims"] = best_params.get(
                    "hidden_dims", self.config["model"]["hidden_dims"]
                )
                self.config["model"]["dropout_rate"] = best_params.get(
                    "dropout", self.config["model"]["dropout_rate"]
                )
            self.model = instantiate_model(self.config["model"], input_dim)
            self.model.to(self.device)

        return best_params

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, use_sampler=False):
    """Create DataLoaders with optional weighted sampling."""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    if use_sampler:
        # Calculate sample weights for imbalanced data
        class_counts = torch.bincount(y_train.long())
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_train.long()]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def log_model_config_params(config):
    """Log most important config params for model comparison"""
    model_type = config["model"]["type"]

    # Essential params to log for comparison
    essential_params = {
        "model_type": model_type,
        "input_dim": config["model"].get("input_dim", "auto"),
        "tuning": config["tuning"],
        "tuning_enabled": config["tuning"].get("enabled", False)
    }

    # Random Forest params
    if model_type == "rf" and "rf_params" in config["model"]:
        rf_config = config["model"]["rf_params"]
        for key, value in rf_config.items():
            if value is not None:
                essential_params[f"rf_{key}"] = value

    # XG Boost params
    elif model_type == "xgb" and "xgb_params" in config["model"]:
        xgb_config = config["model"]["xgb_params"]
        for key, value in xgb_config.items():
            if value is not None:
                essential_params[f"xgb_{key}"] = value

    # Neural Network params
    elif model_type in ["nn_basic", "nn_improved", "logistic"]:
        # Model architecture params
        essential_params.update({
            "nn_hidden_dims": str(config["model"].get("hidden_dims", [])),
            "nn_dropout": config["model"].get("dropout_rate"),
            "nn_activation": config["model"].get("activation"),
            "nn_batch_norm": config["model"].get("use_batch_norm"),
            "nn_residual": config["model"].get("use_residual", False),
        })

        # Training params (only relevant for NNs)
        essential_params.update({
            "epochs": config["training"].get("epochs"),
            "learning_rate": config["training"].get("lr"),
            "batch_size": config["training"].get("batch_size"),
            "optimizer": config["training"].get("optimizer_name"),
            "weight_decay": config["training"].get("weight_decay"),
            "loss_type": config["training"].get("loss_type"),
        })

    # Remove None values
    essential_params = {k: v for k, v in essential_params.items() if v is not None}
    mlflow.log_params(essential_params)
    logger.debug(f"Logged {len(essential_params)} essential params for {model_type.upper()}")
    return essential_params

def main(config_path):
    start_time = time.time()
    with open(config_path, "r") as f:
        config = yaml.load(f)

    experiment_name = config['training'].get('experiment_name')
    logger.debug(f"Experiment Name: {experiment_name}")
    mlflow.set_experiment(f"{experiment_name}")

    run_name = config["training"].get("run_name", "tree_run")

    with mlflow.start_run(run_name=run_name) as run:
        log_model_config_params(config)

        mlflow.log_artifact(config_path)

        # Log model type
        model_type = config["model"]["type"]
        mlflow.set_tag("model_type", model_type)
        logger.info(f"Training {model_type.upper()} model")

        # Extract save_dir from config
        save_dir = config.get("preprocessing", {}).get(
            "save_dir", "experiments/preprocessing/artifacts"
        )
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Using save_dir: {save_dir}")

        # Load data
        state_file = config["data"]["filepath"]["state"]
        train_file = config["data"]["filepath"]["train"]
        val_file = config["data"]["filepath"]["val"]
        test_file = config["data"]["filepath"]["test"]

        # Instantiate and load state once
        preprocessor = DataPreprocessor(save_dir=save_dir)
        preprocessor.load_state(state_file)

        X_train, y_train, y_train_raw = load_dataset(train_file, preprocessor, config)
        X_val, y_val, _ = load_dataset(val_file, preprocessor, config)
        X_test, y_test, _ = load_dataset(test_file, preprocessor, config)

        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Validate input dimensions
        config_input_dim = config['model'].get('input_dim')
        if config_input_dim not in [None, 'auto'] and X_train.shape[1] != config_input_dim:
            logger.warning(
                f"Dataset has {X_train.shape[1]} features, "
                f"but config expects {config_input_dim}"
            )

        # Load feature names
        with open(state_file, "r") as f:
            preprocessor_state = json.load(f)
            feature_names = preprocessor_state["feature_columns"]
        logger.info(f"Loaded {len(feature_names)} feature names")

        # Compute class weights if needed
        if config["training"].get("use_class_weights", False):
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train_raw), y=y_train_raw
            )
            class_weights[1] *= 1.5  # Conservative boost for minority class
            class_weights = np.clip(class_weights, 0.1, 10)
            logger.info(f"Class weights: {class_weights}")
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
        logger.info(f"Model instantiated: {type(model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Trainer
        trainer = ModelTrainer(model, config, device)

        # Set class weights if computed
        if class_weights is not None:
            trainer.class_weights = torch.FloatTensor(class_weights).to(device)

        # Hypertune (if enabled)
        best_params = trainer.hypertune(train_loader, val_loader, input_dim, config)

        # Train
        training_results = trainer.train(
            train_loader, val_loader, config, **config["training"]
        )

        # Get the trained model
        model = trainer.model

        # Log training metrics
        mlflow.log_metric("best_val_loss", training_results["best_val_loss"])
        mlflow.log_metric("epochs_trained", training_results["epochs_trained"])

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
            device=trainer.device,
            save_dir="evaluation_results",
            config_path=config_path,
        )
        metrics, predictions, probabilities, targets = evaluator.evaluate(
            test_loader, feature_names=feature_names
        )

        # Log flat scalar metrics
        serial_metrics = evaluator._convert_to_serializable(metrics)
        flat_metrics = filter_numeric_metrics(serial_metrics)
        mlflow.log_metrics(flat_metrics)

        # === FEATURE IMPORTANCE ===
        logger.info("\n" + "=" * 50)
        logger.info("CALCULATING FEATURE IMPORTANCE")
        logger.info("=" * 50)

        # importance_df = get_feature_importance(
        #     model=model,
        #     X=X_test,
        #     y=y_test,
        #     feature_names=feature_names,
        #     save_dir=save_dir,
        #     model_type="nn",
        #     n_repeats=10
        # )

        # === SAVE RESULTS ===
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M")
        filename = f"final_results_{model_type}_{timestamp}.json"

        results = {
            "date": datetime.now().strftime("%m-%d-%Y %H:%M"),
            "model_type": model_type,
            "config_path": config_path,
            "input_dim": input_dim,
            "best_hyperparams": best_params,
            "training_results": {
                "best_val_loss": training_results["best_val_loss"],
                "epochs_trained": training_results["epochs_trained"],
                "final_train_loss": training_results["train_losses"][-1] if training_results["train_losses"] else None,
                "final_val_loss": training_results["val_losses"][-1] if training_results["val_losses"] else None,
            },
            "test_metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics.get("log_loss"),
            },
            # "top_features": (
            #     importance_df.head(10)[['feature', 'importance']].to_dict('records')
            #     if importance_df is not None else None
            # ),
            "full_config": config,
        }

        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(filepath)
        logger.info(f"Results saved to {filepath}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model", registered_model_name="PredictorModel")
        logger.info("Model logged to MLflow")

        # Log runtime
        end_time = time.time()
        total_runtime = (end_time - start_time) / 60
        mlflow.log_metric("total_runtime_minutes", total_runtime)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("🎉 TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info(f"Model type: {model_type.upper()}")
    logger.info(f"Total runtime: {total_runtime:.2f} minutes")
    logger.info(f"Epochs trained: {training_results['epochs_trained']}")
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    if best_params:
        logger.info(f"Best hyperparameters: {best_params}")
    # if importance_df is not None:
    #     logger.info(f"Top 3 features: {importance_df.head(3)['feature'].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train neural network models (MLP, Logistic Regression)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
