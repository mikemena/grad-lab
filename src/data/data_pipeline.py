import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
import yaml
from data_analyzer import analyze_dataset
from data_preprocessor import DataPreprocessor
from imblearn.over_sampling import SMOTE
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class DataPipeline:
    def __init__(self, save_dir="preprocessing_artifacts"):
        self.preprocessor = DataPreprocessor(save_dir)
        self.column_config = None
        self.save_dir = save_dir

    # SMOTE (Synthetic Minority Oversampling Technique)
    def _detect_class_imbalance(self, y, imbalance_threshold=0.3):
        if isinstance(y, pd.Series):
            y = y.values
        class_counts = pd.Series(y).value_counts(normalize=True)
        minority_proportion = class_counts.min()
        is_imbalanced = minority_proportion < imbalance_threshold
        distribution = {str(k): f"{v:.2%}" for k, v in class_counts.items()}
        logger.info(f"Class distribution before SMOTE: {distribution}")
        return is_imbalanced, distribution

    # SMOTE (Synthetic Minority Oversampling Technique)
    def _apply_smote(self, X, y, random_state=42):
        logger.info("🔄 Applying SMOTE to training data...")
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"✓ SMOTE applied: {X.shape} → {X_resampled.shape}")
        return X_resampled, y_resampled

    def prepare_training_data_with_splits(
        self,
        file_path,
        target_column,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        apply_smote=False,
        imbalance_threshold=0.3,
    ):
        logger.info("=" * 70)
        logger.info("SPLIT EXCEL DATA PIPELINE - MAXIMUM TRANSPARENCY")
        logger.info("=" * 70)
        logger.info(
            f" target column parameter from prepare_training_data_with_splits, {target_column}"
        )
        logger.info("\n1. ANALYZING DATASET STRUCTURE...")
        analysis = analyze_dataset(file_path)
        logger.info(f"✓ Raw data shape: {analysis['shape']}")

        logger.info("\n2. LOADING RAW DATA...")
        df = pd.read_excel(file_path)
        # Reset index to ensure unique indices
        # df = df.reset_index(drop=True)
        # logger.info(f"✓ Loaded: {df.shape}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        logger.info("\n3. CONFIGURING PREPROCESSING...")
        X = df.drop(target_column, axis=1)
        self.column_config = self._extract_column_config(analysis, X.columns)

        logger.info("\n4. CREATING RAW DATA SPLITS...")
        train_df, val_df, test_df = self._split_raw_dataframe(
            df, test_size, val_size, random_state, target_column
        )

        # Debug: Verify split indices
        logger.info("DEBUG: Verifying split indices...")
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        logger.info(f"Train indices: {len(train_indices)} unique")
        logger.info(f"Val indices: {len(val_indices)} unique")
        logger.info(f"Test indices: {len(test_indices)} unique")
        if train_indices & val_indices:
            logger.warning(
                f"Train-Val overlap: {len(train_indices & val_indices)} indices"
            )

        if train_indices & test_indices:
            logger.warning(
                f"Train-Test overlap: {len(train_indices & test_indices)} indices"
            )

        if val_indices & test_indices:
            logger.warning(
                f"Val-Test overlap: {len(val_indices & test_indices)} indices"
            )

        logger.info("\n5. PROCESSING AND SAVING SPLITS...")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        logger.info("🔄 Processing training split...")
        train_excel = f"{base_filename}_train_processed.xlsx"
        X_train_processed = self.preprocessor.process_training_data(
            df=train_df,
            target_column=target_column,
            column_config=self.column_config,
            excel_filename=train_excel,
        )
        y_train = train_df[target_column].values

        logger.info("🔄 Processing validation split...")
        val_excel = f"{base_filename}_val_processed.xlsx"
        X_val_processed = self._process_split_with_target(
            val_df, target_column, val_excel, fit=False
        )
        y_val = val_df[target_column].values

        logger.info("🔄 Processing test split...")
        test_excel = f"{base_filename}_test_processed.xlsx"
        X_test_processed = self._process_split_with_target(
            test_df, target_column, test_excel, fit=False
        )
        y_test = test_df[target_column].values

        if apply_smote:
                    is_imbalanced, distribution = self._detect_class_imbalance(
                        train_df[target_column], imbalance_threshold
                    )
                    if is_imbalanced:
                        X_train_processed, y_train = self._apply_smote(
                            X_train_processed, y_train, random_state
                        )

        logger.info("\n6. CONVERTING TO TENSORS...")
        X_train = torch.FloatTensor(X_train_processed.values)
        X_val = torch.FloatTensor(X_val_processed.values)
        X_test = torch.FloatTensor(X_test_processed.values)
        y_train = torch.FloatTensor(
            y_train
            if self.preprocessor.target_type == "regression"
            else self.preprocessor.target_label_encoder.transform(y_train)
        )
        y_val = torch.FloatTensor(
            y_val
            if self.preprocessor.target_type == "regression"
            else self.preprocessor.target_label_encoder.transform(y_val)
        )
        y_test = torch.FloatTensor(
            y_test
            if self.preprocessor.target_type == "regression"
            else self.preprocessor.target_label_encoder.transform(y_test)
        )

        logger.info(f"✓ Training tensors: {X_train.shape}")
        logger.info(f"✓ Validation tensors: {X_val.shape}")
        logger.info(f"✓ Test tensors: {X_test.shape}")
        logger.info("\n8. PIPELINE SUMMARY")
        logger.info("📁 Generated Files:")
        logger.info(f"📄 {train_excel}")
        logger.info(f"📄 {val_excel}")
        logger.info(f"📄 {test_excel}")

        return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df

    def load_split_data_for_training(self, base_filename):
        logger.info("=" * 70)
        logger.info("LOADING PRE-SPLIT EXCEL FILES FOR TRAINING")
        logger.info("=" * 70)

        train_file = os.path.join(
            self.save_dir, f"{base_filename}_train_processed.xlsx"
        )
        val_file = os.path.join(self.save_dir, f"{base_filename}_val_processed.xlsx")
        test_file = os.path.join(self.save_dir, f"{base_filename}_test_processed.xlsx")

        files_to_check = [train_file, val_file, test_file]
        missing_files = [f for f in files_to_check if not os.path.exists(f)]

        if missing_files:
            logger.error("Missing split files!")
            for file in missing_files:
                logger.error(f"Missing: {file}")
            logger.error("\nRun the split pipeline first to generate these files.")
            raise FileNotFoundError("Split Excel files not found")

        logger.info("✅ All split files found!")
        self.preprocessor.load_state()

        def load_split(file_path, split_name):
            logger.info(f"📄 Loading {split_name}: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path)
            y = df["personal_loan"].values
            X = df.drop("personal_loan", axis=1).values
            logger.info(f"Shape: {X.shape}")
            y = (
                y
                if self.preprocessor.target_type == "regression"
                else self.preprocessor.target_label_encoder.transform(y)
            )
            return torch.FloatTensor(X), torch.FloatTensor(y)

        X_train, y_train = load_split(train_file, "Training")
        X_val, y_val = load_split(val_file, "Validation")
        X_test, y_test = load_split(test_file, "Test")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_raw_dataframe(
        self, df: pd.DataFrame, test_size, val_size, random_state, target_column
    ):
        """Split raw DataFrame into train/val/test before preprocessing"""
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df = df.reset_index(drop=True)
        stratify = (
            df[target_column]
            if not pd.api.types.is_numeric_dtype(df[target_column])
            else None
        )
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_val_df[target_column] if stratify is not None else None,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        return train_df, val_df, test_df

    def _process_split_with_target(self, df, target_column, excel_filename, fit=False):
        self.preprocessor.enable_excel_output(False)
        X = df.drop(target_column, axis=1)
        if self.column_config is None:
            raise ValueError("No column_config provided")
        X_processed = self.preprocessor.process_and_save(
            df=X,
            target_column=None,
            excel_filename=None,
            numerical_columns=self.column_config.get("numerical", []),
            low_cardinality_categorical_columns=self.column_config.get(
                "low_cardinality_categorical", []
            ),
            high_cardinality_categorical_columns=self.column_config.get(
                "high_cardinality_categorical", []
            ),
            binary_columns=self.column_config.get("binary", []),
            datetime_columns=self.column_config.get("datetime", []),
            fit=fit,
        )
        self.preprocessor.enable_excel_output(True)
        full_df = X_processed.copy()
        full_df[target_column] = df[target_column].values
        excel_path = os.path.join(self.save_dir, excel_filename)
        full_df.to_excel(excel_path, index=False)
        logger.info(f"✅ Saved: {excel_filename} ({full_df.shape})")
        return X_processed

    def _extract_column_config(self, analysis, columns):
        config = {
            "datetime": [],
            "binary": [],
            "numerical": [],
            "low_cardinality_categorical": [],
            "high_cardinality_categorical": [],
            "text": [],
        }
        for col in columns:
            col_type = analysis["columns"][col]["recommended_type"]
            if col_type == "datetime":
                config["datetime"].append(col)
            elif col_type == "binary":
                config["binary"].append(col)
            elif col_type == "low_cardinality_categorical":
                config["low_cardinality_categorical"].append(col)
            elif col_type == "high_cardinality_categorical":
                config["high_cardinality_categorical"].append(col)
            elif col_type == "numerical":
                config["numerical"].append(col)
            else:
                config["text"].append(col)
        logger.info("Config:", config)
        return config


def prepare_split_training_data(file_path, target_column="personal_loan", **kwargs):
    pipeline = DataPipeline()
    return pipeline.prepare_training_data_with_splits(
        file_path, target_column, **kwargs
    )


def load_split_training_data(base_filename="loan"):
    pipeline = DataPipeline()
    return pipeline.load_split_data_for_training(base_filename)


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: SPLIT EXCEL PIPELINE")
    logger.info("=" * 70)
    try:
        pipeline = DataPipeline()
        splits = pipeline.prepare_training_data_with_splits(
            "data/bank_loans.xlsx",
            target_column="personal_loan",
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )
        logger.info("\n" + "=" * 70)
        logger.info("DEMO: LOADING SPLIT FILES")
        logger.info("=" * 70)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            pipeline.load_split_data_for_training("loan")
        )
        logger.info("\n Split Excel pipeline complete!")
    except FileNotFoundError:
        logger.error("File not found")
