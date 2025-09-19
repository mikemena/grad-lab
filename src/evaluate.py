import torch
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    matthews_corrcoef,
    log_loss,
)
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)

yaml = YAML()


class ModelEvaluator:
    def __init__(
        self, model=None, device=None, save_dir="evaluation_results", config_path=None
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.metrics_history = []
        self.save_dir = save_dir
        self.config_path = config_path
        self.yaml = yaml
        os.makedirs(self.save_dir, exist_ok=True)
        if config_path:
            logger.info(f"ModelEvaluator initialized with config_path: {config_path}")
        else:
            logger.warning("No config_path provided; config file will not be updated.")

    def _convert_to_serializable(self, obj):
        """Recursively convert non-serializable objects to JSON-serializable types."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    def evaluate(self, test_loader, feature_names=None):
        """Evaluate model on test set with comprehensive metrics"""
        if self.model is None:
            logger.error("No model provided for evaluation")
            raise ValueError("Model must be provided for evaluation")

        self.model.eval()
        predictions = []
        probabilities = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits)
                probabilities.extend(probs.cpu().numpy())
                predictions.extend((probs >= 0.5).float().cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        probabilities = np.array(probabilities).flatten()
        targets = np.array(targets).flatten()

        # Check if target type is consistent with binary classification
        if not np.all(np.isin(targets, [0, 1])):
            logger.warning(
                "Target values contain non-binary values; expected binary classification targets (0 or 1)."
            )

        # Compute comprehensive metrics with default threshold (0.5)
        metrics = self.comprehensive_evaluation(targets, probabilities, predictions)

        # Optimize threshold for recall
        optimal_recall_threshold, optimal_recall_score, thresholds, scores = (
            self.optimize_threshold(targets, probabilities, metric="recall")
        )
        metrics["optimal_recall_threshold"] = optimal_recall_threshold
        metrics["optimal_recall_score"] = optimal_recall_score

        # Recompute metrics with optimal recall threshold
        y_pred_new = (probabilities >= optimal_recall_threshold).astype(int)
        new_metrics = self.comprehensive_evaluation(targets, probabilities, y_pred_new)
        logger.info(
            f"Metrics with optimal recall threshold ({optimal_recall_threshold:.2f}):"
        )
        logger.info(
            f"Recall: {new_metrics['recall']:.4f}, Precision: {new_metrics['precision']:.4f}, F1: {new_metrics['f1']:.4f}"
        )

        # Perform business impact analysis
        business_analyzer = BusinessImpactAnalyzer(self.config_path)
        business_metrics = business_analyzer.calculate_business_value(
            targets, predictions
        )
        optimal_threshold, optimal_business_value = (
            business_analyzer.optimize_for_business_value(targets, probabilities)
        )
        metrics.update(business_metrics)
        metrics["optimal_business_threshold"] = optimal_threshold
        metrics["optimal_business_value"] = optimal_business_value

        # Generate visualizations
        self.plot_evaluation_metrics(metrics, targets, probabilities, feature_names)

        # Save results
        self.save_evaluation_results(metrics, targets, probabilities)

        logger.info("Test Set Evaluation:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric_name}: {value:.4f}")
            else:
                logger.info(f"{metric_name}: {value}")

        return metrics, predictions, probabilities, targets

    def comprehensive_evaluation(
        self, y_true, y_pred_proba, y_pred_binary, threshold=0.5
    ):
        """Compute comprehensive evaluation metrics"""
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred_binary),
                "precision": precision_score(y_true, y_pred_binary, zero_division=0),
                "recall": recall_score(y_true, y_pred_binary, zero_division=0),
                "f1": f1_score(y_true, y_pred_binary, zero_division=0),
                "f2": fbeta_score(y_true, y_pred_binary, beta=2, zero_division=0),
                "matthews_corrcoef": matthews_corrcoef(y_true, y_pred_binary),
                "log_loss": log_loss(y_true, y_pred_proba),
                "average_precision": average_precision_score(y_true, y_pred_proba),
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "precision_at_k": self.precision_at_k(y_true, y_pred_proba, k=100),
                "lift_at_k": self.lift_at_k(y_true, y_pred_proba, k=100),
            }

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            metrics.update(
                {
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                    "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                    "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0 else 0,
                    "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
                }
            )

            metrics["classification_report"] = classification_report(
                y_true, y_pred_binary, output_dict=True, zero_division=0
            )

            return metrics
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {str(e)}")
            raise

    def precision_at_k(self, y_true, y_scores, k):
        """Precision at top k predictions"""
        top_k_indices = np.argsort(y_scores)[-k:]
        return np.mean(y_true[top_k_indices])

    def lift_at_k(self, y_true, y_scores, k):
        """Lift at top k predictions"""
        precision_at_k = self.precision_at_k(y_true, y_scores, k)
        baseline_precision = np.mean(y_true)
        return precision_at_k / baseline_precision if baseline_precision > 0 else 0

    def optimize_threshold(self, y_true, y_pred_proba, metric="recall"):
        """Find optimal threshold for binary classification"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "f2":
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "matthews":
                score = matthews_corrcoef(y_true, y_pred)
            else:
                logger.error(f"Unsupported metric for threshold optimization: {metric}")
                raise ValueError(f"Unsupported metric: {metric}")
            scores.append(score)

        optimal_idx = np.argmax(scores)
        return thresholds[optimal_idx], scores[optimal_idx], thresholds, scores

    def compute_data_drift(self, X_train, X_new, feature_names):
        """Detect data drift using statistical tests"""
        drift_results = {}
        for i, feature in enumerate(feature_names):
            train_feature = X_train[:, i]
            new_feature = X_new[:, i]
            ks_stat, ks_p_value = ks_2samp(train_feature, new_feature)
            drift_results[feature] = {
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p_value,
                "drift_detected": ks_p_value < 0.05,
                "drift_magnitude": ks_stat,
            }
        return drift_results

    def monitor_performance_degradation(
        self, current_metrics, baseline_metrics, threshold=0.05
    ):
        """Check if model performance has degraded significantly"""
        degradation_alerts = {}
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics and isinstance(current_value, (int, float)):
                baseline_value = baseline_metrics[metric]
                degradation = (
                    (baseline_value - current_value) / baseline_value
                    if baseline_value > 0
                    else 0
                )
                degradation_alerts[metric] = {
                    "current_value": current_value,
                    "baseline_value": baseline_value,
                    "degradation_pct": degradation * 100,
                    "alert": degradation > threshold,
                }
        return degradation_alerts

    def plot_evaluation_metrics(self, metrics, targets, probabilities, feature_names):
        """Generate and save evaluation plots"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(targets, (probabilities >= 0.5).astype(int))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"))
            plt.close()

            fpr, tpr, _ = roc_curve(targets, probabilities)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.4f}')
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, "roc_curve.png"))
            plt.close()

            precision, recall, thresholds = precision_recall_curve(
                targets, probabilities
            )
            plt.figure(figsize=(8, 6))
            plt.plot(
                recall, precision, label=f'AP = {metrics["average_precision"]:.4f}'
            )
            optimal_recall_threshold = metrics.get("optimal_recall_threshold", 0.5)
            idx = np.argmin(np.abs(thresholds - optimal_recall_threshold))
            plt.scatter(
                recall[idx],
                precision[idx],
                color="red",
                s=100,
                label=f"Optimal Recall Threshold = {optimal_recall_threshold:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "precision_recall_curve.png"))
            plt.close()

            metric_names = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f2",
                "matthews_corrcoef",
            ]
            values = [metrics.get(m, 0) for m in metric_names]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=metric_names, y=values)
            plt.title("Classification Metrics")
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(self.save_dir, "metrics_bar.png"))
            plt.close()

        except Exception as e:
            logger.error(f"Error in plotting evaluation metrics: {str(e)}")

    def save_evaluation_results(self, metrics, targets, probabilities):
        """Save evaluation results to JSON and update config with optimal threshold"""
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M")
        filename = f"eval_results_{timestamp}.json"
        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "metrics": self._convert_to_serializable(metrics),
            "predictions": self._convert_to_serializable(probabilities),
            "targets": self._convert_to_serializable(targets),
        }
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(
            f"Evaluation results saved to {self.save_dir}/evaluation_results.json"
        )

        if self.config_path:
            try:
                logger.debug(f"Metrics dictionary keys: {list(metrics.keys())}")
                logger.debug(f"Attempting to update config file: {self.config_path}")
                if "optimal_recall_threshold" not in metrics:
                    logger.error("Key 'optimal_recall_threshold' not found in metrics")
                    raise KeyError("optimal_recall_threshold")

                self.yaml.preserve_quotes = True
                # Force flow style for sequences (lists/arrays)
                self.yaml.default_flow_style = None  # Preserve original style

                with open(self.config_path, "r") as f:
                    config = self.yaml.load(f)

                if not config:
                    logger.error(f"Config file {self.config_path} is empty or invalid")
                    raise ValueError(f"Invalid config file: {self.config_path}")
                if "inference" not in config:
                    logger.error("No 'inference' section in config file")
                    raise KeyError("inference")
                if "decision_threshold" not in config["inference"]:
                    logger.error("No 'decision_threshold' key in config['inference']")
                    raise KeyError("decision_threshold")

                config["inference"]["decision_threshold"] = float(
                    metrics["optimal_business_threshold"]
                )
                with open(self.config_path, "w") as f:
                    self.yaml.dump(config, f)

                logger.info(
                    f"Updated config {self.config_path} with optimal business threshold: {metrics['optimal_business_threshold']:.2f}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to update config file {self.config_path}: {str(e)}"
                )
                raise
        else:
            logger.warning("No config_path provided; skipping config update")

    def plot_training_history(
        self,
        train_losses,
        val_losses,
        display=True,
        save=True,
        filename="training_history.png",
    ):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (BCE)")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_data_distribution(
        self,
        data,
        column_names,
        max_cols=3,
        display=True,
        save=True,
        filename="data_distribution.png",
    ):
        """Plot histogram of numerical features to show data distribution with improved readability"""
        df = pd.DataFrame(data, columns=column_names)
        n_features = len(column_names)
        n_rows = (
            n_features + max_cols - 1
        ) // max_cols  # Dynamic rows based on features

        plt.figure(figsize=(15, 5 * n_rows))  # Adjust height based on number of rows
        for idx, column in enumerate(column_names):
            plt.subplot(n_rows, max_cols, idx + 1)
            sns.histplot(df[column], bins=30, kde=True)
            plt.title(f"Distribution of {column}", fontsize=10)
            plt.xlabel(column, rotation=45)
            plt.ylabel("Count", fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

        plt.tight_layout(pad=2.0)  # Increase padding to avoid overlap

        if save:
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight"
            )
        if display:
            plt.show()
        plt.close()

    def plot_prediction_distribution(
        self,
        predictions,
        targets,
        display=True,
        save=True,
        filename="prediction_distribution.png",
    ):
        """Plot distribution of predicted probabilities by class"""
        plt.figure(figsize=(8, 6))
        plt.hist(
            predictions[targets == 0], bins=20, alpha=0.5, label="Class 0", color="blue"
        )
        plt.hist(
            predictions[targets == 1],
            bins=20,
            alpha=0.5,
            label="Class 1",
            color="orange",
        )
        plt.xlabel("Predicted Probabilities")
        plt.ylabel("Count")
        plt.title("Distribution of Predicted Probabilities by Class")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def visualize_original_vs_synthetic_samples():
        original_train = pd.read_excel("debug_splits/raw_train_split.xlsx")
        resampled_train = pd.read_excel(
            "preprocessing_artifacts/loan_train_resampled.xlsx"
        )
        X_original = original_train.drop("personal_loan", axis=1)
        y_original = original_train["personal_loan"]
        X_resampled = resampled_train.drop("personal_loan", axis=1)
        y_resampled = resampled_train["personal_loan"]

        pca = PCA(n_components=2)
        X_original_pca = pca.fit_transform(X_original)
        X_resampled_pca = pca.transform(X_resampled)

        plt.scatter(
            X_original_pca[:, 0],
            X_original_pca[:, 1],
            c=y_original,
            label="Original",
            alpha=0.5,
        )
        plt.scatter(
            X_resampled_pca[:, 0],
            X_resampled_pca[:, 1],
            c=y_resampled,
            label="Resampled",
            alpha=0.2,
        )
        plt.legend()
        plt.title("PCA of Original vs. Resampled Data")
        plt.show()


class BusinessImpactAnalyzer:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.yaml = yaml

    def calculate_business_value(self, y_true, y_pred):
        """Calculate business value based on confusion matrix"""
        if self.config_path is None:
            raise ValueError(
                "config_path must be provided to load business costs/benefits."
            )
        with open(self.config_path, "r") as f:
            config = self.yaml.load(f)

        cost_fp = config["inference"]["cost_false_positives"]
        cost_fn = config["inference"]["cost_false_negatives"]
        benefit_tp = config["inference"]["benefit_true_positives"]
        logger.info(f"Cost of false positives: {cost_fp}")
        logger.info(f"Cost of false negatives: {cost_fn}")
        logger.info(f"Benefit of true positive: {benefit_tp}")

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_value = (
                tp * benefit_tp  # Revenue from true positives
                - fp * cost_fp  # Cost of false positives
                - fn * cost_fn  # Cost of false negatives
            )
            logger.info(f"Total Value: {total_value}")
            return {
                "total_business_value": float(total_value),
                "value_per_prediction": (
                    float(total_value) / len(y_true) if len(y_true) > 0 else 0
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        except Exception as e:
            logger.error(f"Error in business value calculation: {str(e)}")
            raise

    def optimize_for_business_value(self, y_true, y_pred_proba):
        """Find threshold that maximizes business value"""
        try:
            thresholds = np.arange(0.1, 0.9, 0.01)
            business_values = []
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                value = self.calculate_business_value(y_true, y_pred)[
                    "total_business_value"
                ]
                business_values.append(value)
            optimal_idx = np.argmax(business_values)
            return thresholds[optimal_idx], business_values[optimal_idx]
        except Exception as e:
            logger.error(f"Error in business value optimization: {str(e)}")
            raise


if __name__ == "__main__":
    logger.info("This module is intended to be imported and used with a trained model.")
