import torch
import os
import sys
import argparse
import yaml
from data_pipeline import DataPipeline
import pandas as pd
from hashlib import sha1

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


def prepare_training_data(config):
    """Prepare and save training data with both processed and raw splits"""
    logger.info("Starting data preparation...")
    save_dir = config.get("preprocessing", {}).get(
        "save_dir", "preprocessing_artifacts"
    )
    config = config
    logger.debug(f"Config: {config}")

    logger.debug(f"save_dir from config: {save_dir}")
    pipeline = DataPipeline(save_dir=save_dir, config=config)

    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = (
        pipeline.prepare_training_data_with_splits(
            config["file_path"],
            target_column=config["target_column"],
            drop_columns=config["drop_columns"],
            test_size=config["splits"]["test_size"],
            val_size=config["splits"]["val_size"],
            random_state=config["random_state"],
            apply_smote=config["imbalance"]["apply_smote"],
            imbalance_threshold=config["imbalance"]["threshold"],
        )
    )
    logger.info(
        f"With SMOTE - Training class distribution: {pd.Series(y_train.numpy()).value_counts(normalize=True)}"
    )
    logger.info("\nData preparation complete!")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Save processed tensors if configured
    if config.get("output", {}).get("save_processed_data", True):
        torch.save(
            {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            },
            os.path.join(save_dir, "processed_data.pt"),
        )
        logger.info(
            f"✅ Processed tensors saved to '{os.path.join(save_dir, 'processed_data.pt')}'"
        )
    # Save debug splits if configured
    if config.get("output", {}).get("debug_splits", True):
        logger.info("\n🔍 Saving raw dataframes for debugging...")
        debug_dir = config["output"]["debug_splits_dir"]

        # Create debug directory if it doesn't exist
        # os.makedirs(debug_dir, exist_ok=True)

        train_df.to_excel(os.path.join(debug_dir, "raw_train_split.xlsx"), index=False)
        val_df.to_excel(os.path.join(debug_dir, "raw_val_split.xlsx"), index=False)
        test_df.to_excel(os.path.join(debug_dir, "raw_test_split.xlsx"), index=False)

        logger.info(f"✅ Raw splits saved to '{debug_dir}/' directory:")
        logger.info(f"📄 raw_train_split.xlsx ({train_df.shape})")
        logger.info(f"📄 raw_val_split.xlsx ({val_df.shape})")
        logger.info(f"📄 raw_test_split.xlsx ({test_df.shape})")

        create_split_summary(
            train_df,
            val_df,
            test_df,
            debug_dir,
            target_column=config["target_column"],
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_split_summary(train_df, val_df, test_df, debug_dir, target_column):
    """Create a summary Excel file for quick split analysis"""
    # Calculate total rows for percentage calculation
    total_rows = len(train_df) + len(val_df) + len(test_df)

    summary_data = []
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        summary = {
            "Split": split_name,
            "Count": len(df),
            "Percentage": f"{len(df)/total_rows*100:.1f}%",
            "Missing_Values": df.isnull().sum().sum(),
        }
        if pd.api.types.is_numeric_dtype(df[target_column]):
            summary.update(
                {
                    "Target_Mean": f"{df[target_column].mean():.2f}",
                    "Target_Std": f"{df[target_column].std():.2f}",
                    "Target_Min": df[target_column].min(),
                    "Target_Max": df[target_column].max(),
                }
            )
        else:
            value_counts = df[target_column].value_counts(normalize=True)
            summary.update(
                {
                    "Target_Distribution": ", ".join(
                        [f"{k}: {v:.2%}" for k, v in value_counts.items()]
                    )
                }
            )
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(debug_dir, "split_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    logger.info(f"✅ Split summary saved to '{summary_path}'")
    logger.info("\n📊 Quick Summary:")
    for _, row in summary_df.iterrows():
        logger.info(f"{row['Split']}: {row['Count']} samples ({row['Percentage']})")
        if "Target_Mean" in row:
            logger.info(f"Target avg: {row['Target_Mean']}, std: {row['Target_Std']}")
        else:
            logger.info(f"Target distribution: {row['Target_Distribution']}")


def debug_splits(config):
    """Function to help debug splitting issues after training"""
    logger.info("🔍 DEBUGGING SPLIT INTEGRITY...")

    # Get debug directory from config
    debug_dir = config.get("output", {}).get("debug_splits_dir", "debug_splits")
    logger.debug(f"debug_dir from config: {debug_dir}")
    target_column = config["target_column"]

    try:
        train_df = pd.read_excel(os.path.join(debug_dir, "raw_train_split.xlsx"))
        val_df = pd.read_excel(os.path.join(debug_dir, "raw_val_split.xlsx"))
        test_df = pd.read_excel(os.path.join(debug_dir, "raw_test_split.xlsx"))

        logger.info("✅ Raw split files loaded successfully")

        # Use hashes as unique identifiers to prevent data leakage detection
        def hash_rows(df):
            return set(
                df.astype(str).apply(
                    lambda row: sha1("||".join(row).encode()).hexdigest(), axis=1
                )
            )

        train_hashes = hash_rows(train_df)
        val_hashes = hash_rows(val_df)
        test_hashes = hash_rows(test_df)

        overlaps = []
        if train_hashes & val_hashes:
            overlaps.append(
                f"Train-Val overlap detected! ({len(train_hashes & val_hashes)} rows)"
            )
        if train_hashes & test_hashes:
            overlaps.append(
                f"Train-Test overlap detected! ({len(train_hashes & test_hashes)} rows)"
            )
        if val_hashes & test_hashes:
            overlaps.append(
                f"Val-Test overlap detected! ({len(val_hashes & test_hashes)} rows)"
            )

        if overlaps:
            logger.info("❌ DATA LEAKAGE DETECTED:")
            for msg in overlaps:
                logger.info(f"{msg}")
        else:
            logger.info("✅ No data leakage detected")

        logger.info("\n📊 Target Distribution Check:")
        if pd.api.types.is_numeric_dtype(train_df[target_column]):
            logger.info(
                f"Train {target_column} range: {train_df[target_column].min()}-{train_df[target_column].max()}"
            )
            logger.info(
                f"Val {target_column} range: {val_df[target_column].min()}-{val_df[target_column].max()}"
            )
            logger.info(
                f"Test {target_column} range: {test_df[target_column].min()}-{test_df[target_column].max()}"
            )

        else:
            for split_name, df in [
                ("Train", train_df),
                ("Val", val_df),
                ("Test", test_df),
            ]:
                value_counts = df[target_column].value_counts(normalize=True)
                logger.info(
                    f"{split_name} {target_column} distribution: {', '.join([f'{k}: {v:.2%}' for k, v in value_counts.items()])}"
                )

    except FileNotFoundError:
        logger.error("❌ Debug files not found. Run prepare_training_data() first.")
    except Exception as e:
        logger.error(f"❌ Error during debugging: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data with config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_v1.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Pass config to the function
    prepare_training_data(config)
    logger.info("\n" + "=" * 50)
    debug_splits(config)


if __name__ == "__main__":
    main()
