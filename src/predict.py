import os
import torch
import yaml
import argparse
import pandas as pd
from train_model import instantiate_model
from data.data_preprocessor import DataPreprocessor
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def load_unseen_data(file_path, preprocessor, config):
    """Preprocess unseen data (no labels assumed)."""
    df = pd.read_excel(file_path)  # Or pd.read_csv if CSV
    # Drop any irrelevant columns if needed (match training)
    X = df.values  # Assuming all columns are features
    X_preprocessed = preprocessor.transform(X)  # Apply saved transformations
    X_tensor = torch.tensor(X_preprocessed, dtype=torch.float32)
    return X_tensor, df  # Return original df for output merging


def main(config_path, input_path, output_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract save_dir from config
    save_dir = config["preprocessing"]["save_dir"]
    logger.info(f"Using save_dir: {save_dir}")
    state_file = config["data"]["filepath"]["state"]
    logger.info(f"Using state_dir: {state_file}")

    # Use input_path if provided, otherwise fall back to config test path
    if input_path is None:
        input_path = config["data"]["filepath"]["test"]
        logger.info(f"Using input_path: {input_path}")
    else:
        logger.info(f"Using provided input path: {input_path}")

    # Load preprocessor and model
    preprocessor = DataPreprocessor(save_dir=save_dir)
    preprocessor.load_state(state_file)

    # Load and preprocess unseen data
    X_unseen, original_df = load_unseen_data(input_path, preprocessor, config)
    input_dim = X_unseen.shape[1]

    # Instantiate model
    model = instantiate_model(config["model"], input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load best model weights
    save_path = os.path.join(save_dir, "best_model.pt")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(X_unseen.to(device))
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)  # Binary threshold

    # Prepare output
    original_df["probability"] = probabilities
    original_df["prediction"] = predictions
    if not output_path:
        output_path = os.path.join(save_dir, "predictions.csv")
    original_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to unseen data file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save predictions (optional)"
    )
    args = parser.parse_args()
    main(args.config, args.input, args.output)
