import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
import os
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class DataPreprocessor:
    """Handles all data preprocessing with state management and Excel output"""

    def __init__(self, save_dir="preprocessing_artifacts"):
        self.save_dir = save_dir
        self.scaler = None
        self.feature_columns = None
        self.column_mappings = {}
        self.target_encoding_mappings = {}
        self.target_label_encoder = None  # For categorical targets
        self.target_type = None  # Store target type (regression, binary, categorical)
        self.excel_output_enabled = True
        self.one_hot_encoder = None  # For categorical and binary columns
        os.makedirs(save_dir, exist_ok=True)

    def preprocess_datetime(self, df, date_columns):
        """Extract features from datetime columns"""
        df = df.copy()
        for col in date_columns:
            if col in df.columns:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
                df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
                df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
                df = df.drop(col, axis=1)
        return df

    def preprocess_categorical(self, df, categorical_columns, binary_columns=None):
        """Handle categorical and binary variables using OneHotEncoder"""
        df = df.copy()
        categorical_to_encode = [
            col
            for col in (categorical_columns or []) + (binary_columns or [])
            if col in df.columns
        ]

        if not categorical_to_encode:
            return df

        logger.info(
            f"Processing {len(categorical_to_encode)} categorical/binary columns with OneHotEncoder..."
        )

        # Convert columns to string to handle mixed types and ensure consistency
        for col in categorical_to_encode:
            df[col] = df[col].astype(str).fillna("Unknown")

        if self.one_hot_encoder is None:
            # Initialize encoder during fit
            self.one_hot_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", dtype=np.int32
            )
            encoded_data = self.one_hot_encoder.fit_transform(df[categorical_to_encode])
        else:
            # Use existing encoder for transform
            encoded_data = self.one_hot_encoder.transform(df[categorical_to_encode])

        # Get feature names from encoder
        encoded_columns = self.one_hot_encoder.get_feature_names_out(
            categorical_to_encode
        )

        # Create DataFrame with encoded columns
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)

        # Drop original categorical columns and concatenate encoded columns
        df = df.drop(categorical_to_encode, axis=1)
        df = pd.concat([df, encoded_df], axis=1)

        return df

    def determine_target_type(self, series, unique_count, total_rows):
        """Determine the target column type"""
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_proportion = unique_count / total_rows

        if is_numeric:
            return "regression"
        elif unique_count == 2:
            return "binary"
        elif unique_count <= 5 and not is_numeric:
            return "categorical"
        elif unique_count >= 6 and not is_numeric:
            if unique_proportion > 0.9:
                return "text"  # Unlikely for target, but included for completeness
            return "categorical"
        return "text"

    def target_encode(self, df, feature_column, target_data=None, alpha=0.1, fit=True):
        """Transform a high-cardinality categorical feature into a numerical feature"""
        logger.info(
            f"Debugging target_encode for feature: {feature_column} (fit={fit})"
        )
        df = df.copy()

        if fit:
            if target_data is None:
                raise ValueError("Target data must be provided when fit=True")
            if not isinstance(target_data, pd.Series):
                target_data = pd.Series(target_data, index=df.index)
            if target_data.isna().any():
                raise ValueError(
                    f"Target data contains {target_data.isna().sum()} NaNs"
                )
            global_mean = target_data.mean()
            logger.info(f"Global mean: {global_mean}")
            temp_df = pd.DataFrame(
                {"feature": df[feature_column].astype(str), "target": target_data}
            )
            mapping = (
                temp_df.groupby("feature")["target"]
                .apply(lambda x: (x.sum() + alpha * global_mean) / (x.count() + alpha))
                .to_dict()
            )
            # logger.info(f"Target encoding mapping: {mapping}")
            self.target_encoding_mappings[feature_column] = {
                "mapping": mapping,
                "global_mean": float(global_mean),
            }
            df[f"{feature_column}_encoded"] = (
                df[feature_column].astype(str).map(mapping)
            )
            df[f"{feature_column}_encoded"] = df[f"{feature_column}_encoded"].fillna(
                global_mean
            )
            if df[f"{feature_column}_encoded"].isna().any():
                logger.warning(f"NaNs detected in {feature_column}_encoded")
        else:
            if feature_column not in self.target_encoding_mappings:
                logger.error(
                    f"No target encoding mapping found for {feature_column}. Run fit=True first."
                )
                raise ValueError(
                    f"No target encoding mapping found for {feature_column}. Run fit=True first."
                )
            mapping_info = self.target_encoding_mappings[feature_column]
            mapping = mapping_info["mapping"]
            global_mean = mapping_info["global_mean"]
            df[f"{feature_column}_encoded"] = (
                df[feature_column].astype(str).map(mapping)
            )
            df[f"{feature_column}_encoded"] = df[f"{feature_column}_encoded"].fillna(
                global_mean
            )

        return df.drop(feature_column, axis=1)

    def handle_missing_values(self, df, numerical_columns, categorical_columns):
        """Fill missing values appropriately"""
        df = df.copy()
        for col in numerical_columns:
            if col in df.columns and df[col].isnull().any():
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")
        return df

    def scale_features(self, df, fit=True):
        """Scale features using StandardScaler"""
        if fit:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            if self.scaler is None:
                logger.error("Scaler not fitted. Set fit=True or load saved scaler.")
                raise ValueError(
                    "Scaler not fitted. Set fit=True or load saved scaler."
                )
            scaled_data = self.scaler.transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    def align_features(self, df, fit=True):
        """Ensure feature alignment between training and prediction"""
        if fit:
            self.feature_columns = df.columns.tolist()
        else:
            if self.feature_columns is None:
                logger.error(
                    "Feature columns not set. Set fit=True or load saved state."
                )
                raise ValueError(
                    "Feature columns not set. Set fit=True or load saved state."
                )
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        return df

    def process_and_save(
        self,
        df,
        target_column=None,
        excel_filename=None,
        numerical_columns=None,
        low_cardinality_categorical_columns=None,
        high_cardinality_categorical_columns=None,
        binary_columns=None,
        datetime_columns=None,
        fit=True,
    ):

        logger.info(f"🔄 Starting preprocessing pipeline (fit={fit})...")

        # Separate target if provided
        if target_column and target_column in df.columns:
            target_data = df[target_column].copy()
            logger.info(f"Target data dtype: {target_data.dtype}")
            unique_targets = target_data.unique()
            unique_count = len(unique_targets)
            total_rows = len(target_data)
            logger.info(f"Unique target values: {unique_targets}")

            # Determine target type
            self.target_type = self.determine_target_type(
                target_data, unique_count, total_rows
            )
            logger.info(f"Detected target type: {self.target_type}")

            # Encode categorical targets
            if self.target_type in ["binary", "categorical"] and fit:
                self.target_label_encoder = LabelEncoder()
                target_data_encoded = pd.Series(
                    self.target_label_encoder.fit_transform(target_data),
                    index=target_data.index,
                )
                logger.info(f"Encoded target values: {target_data_encoded.unique()}")
            elif self.target_type in ["binary", "categorical"] and not fit:
                if self.target_label_encoder is None:
                    logger.error("Target label encoder not fitted. Run fit=True first.")
                    raise ValueError(
                        "Target label encoder not fitted. Run fit=True first."
                    )
                target_data_encoded = pd.Series(
                    self.target_label_encoder.transform(target_data),
                    index=target_data.index,
                )
            else:
                target_data_encoded = (
                    target_data  # Regression target, no encoding needed
                )

            if target_data_encoded.isna().any():
                logger.error(
                    f"Target data contains {target_data_encoded.isna().sum()} NaNs after encoding"
                )
                raise ValueError(
                    f"Target data contains {target_data_encoded.isna().sum()} NaNs after encoding"
                )

            features_df = df.drop(target_column, axis=1).copy()
            logger.info(f"✓ Separated target column '{target_column}'")
        else:
            target_data = None
            target_data_encoded = None
            features_df = df.copy()

        # Store original shape for reporting
        original_shape = features_df.shape

        # Add temporary index for inference data to ensure row matching, only if not already present
        if not fit and "temp_index" not in features_df.columns:
            features_df["temp_index"] = np.arange(len(features_df))

        # Step 1: Handle missing values
        if numerical_columns or low_cardinality_categorical_columns:
            logger.info("Handling missing values...")
            all_categorical = (low_cardinality_categorical_columns or []) + (
                binary_columns or []
            )
            features_df = self.handle_missing_values(
                features_df, numerical_columns or [], all_categorical
            )

        # Step 2: Process datetime columns
        if datetime_columns:
            logger.info(f"Processing {len(datetime_columns)} datetime columns...")
            features_df = self.preprocess_datetime(features_df, datetime_columns)

        # Step 3: Process low-cardinality categorical and binary features
        if low_cardinality_categorical_columns or binary_columns:
            logger.info("Processing low cardinality categorical/binary columns...")
            features_df = self.preprocess_categorical(
                features_df, low_cardinality_categorical_columns or [], binary_columns
            )

        # Step 4: Process high-cardinality categorical feature
        if high_cardinality_categorical_columns:
            logger.info("Processing high cardinality categorical columns...")
            for col in high_cardinality_categorical_columns:
                if col in features_df.columns:
                    # logger.info(f"Before target encoding for {col}:")
                    # logger.info(features_df.head())
                    features_df = self.target_encode(
                        features_df,
                        col,
                        target_data=target_data_encoded if fit else None,
                        fit=fit,
                    )
                    # logger.info(f"After target encoding for {col}:")
                    if features_df[f"{col}_encoded"].isna().any():
                        logger.warning(f"NaNs detected in {col}_encoded")
                        features_df.to_excel(
                            os.path.join(
                                self.save_dir, f"after_target_encode_{col}.xlsx"
                            ),
                            index=False,
                        )

        # Debug: Check for NaNs or invalid values before scaling
        logger.info("Debug: Checking data before scaling...")
        nan_columns = features_df.columns[features_df.isna().any()].tolist()
        inf_columns = features_df.columns[
            features_df.isin([np.inf, -np.inf]).any()
        ].tolist()
        logger.info(f"NaN columns: {nan_columns}")
        logger.info(f"Infinity columns: {inf_columns}")
        if nan_columns or inf_columns:
            features_df.to_excel(
                os.path.join(self.save_dir, "before_scaling_debug.xlsx"), index=False
            )
            logger.info("Saved pre-scaling DataFrame to before_scaling_debug.xlsx")

        # Step 5: Align features and scale, excluding temp_index if present
        logger.info("🔄 Aligning features...")
        columns_to_process = [col for col in features_df.columns if col != "temp_index"]
        features_to_process = features_df[columns_to_process]
        features_to_process = self.align_features(features_to_process, fit=fit)

        logger.info("📊 Scaling features...")
        features_to_process = self.scale_features(features_to_process, fit=fit)

        if not fit:
            features_to_process["temp_index"] = features_df["temp_index"]

        features_df = features_to_process
        logger.info(
            f"Scaler n_samples_seen_: {self.scaler.n_samples_seen_ if self.scaler else 'N/A'}"
        )

        logger.info(f"✓ Preprocessing complete: {original_shape} → {features_df.shape}")

        # Debug: Check for NaNs after scaling
        if features_df.isna().any().any():
            logger.warning("NaNs detected after scaling")
            logger.info(features_df.isna().sum())
            features_df.to_excel(
                os.path.join(self.save_dir, "after_scaling_debug.xlsx"), index=False
            )

        # Step 7: Save to Excel if enabled
        if self.excel_output_enabled:
            self._save_to_excel(
                features_df, target_data, target_column, excel_filename, fit
            )

        # Step 8: Save preprocessing state if fitting
        if fit:
            logger.info("💾 Saving preprocessing state...")
            self.save_state()
            logger.info(
                f"✓ State saved to {os.path.join(self.save_dir, 'preprocessor_state.json')}"
            )

        return features_df

    def _save_to_excel(
        self, features_df, target_data, target_column, excel_filename, fit
    ):
        """Internal method to save processed data to Excel"""
        if excel_filename is None:
            suffix = "processed" if fit else "inference_processed"
            excel_filename = f"data_{suffix}.xlsx"

        excel_path = os.path.join(self.save_dir, excel_filename)

        # Combine features and target for Excel output
        if target_data is not None:
            excel_df = features_df.copy()
            excel_df[target_column] = (
                target_data.values
            )  # Use original target_data (not encoded)
            logger.info("📄 Saving processed data with target to Excel...")
        else:
            excel_df = features_df.copy()
            logger.info("📄 Saving processed features to Excel...")

        excel_df.to_excel(excel_path, index=False)
        logger.info(f"✅ Excel saved: {excel_path}")
        logger.info(f"Shape: {excel_df.shape}")
        logger.info(
            f"Columns: {len(excel_df.columns)} ({list(excel_df.columns)[:3]}...)"
        )

        return excel_path

    def enable_excel_output(self, enabled=True):
        """Enable or disable Excel output"""
        self.excel_output_enabled = enabled
        logger.info(f"Excel output {'enabled' if enabled else 'disabled'}")

    def save_state(self):
        """Save preprocessing state as JSON"""
        if self.scaler is None:
            logger.error("Scaler not fitted. Cannot save state.")
            raise ValueError("Scaler not fitted. Cannot save state.")

        # Extract StandardScaler parameters
        n_samples_seen = (
            int(self.scaler.n_samples_seen_[0])
            if isinstance(self.scaler.n_samples_seen_, np.ndarray)
            and self.scaler.n_samples_seen_.size > 0
            else (
                int(self.scaler.n_samples_seen_)
                if not isinstance(self.scaler.n_samples_seen_, np.ndarray)
                else 0
            )
        )
        logger.info(
            f"Scaler n_samples_seen_ raw: {self.scaler.n_samples_seen_}, converted: {n_samples_seen}"
        )

        scaler_params = {
            "mean_": (
                self.scaler.mean_.tolist() if self.scaler.mean_ is not None else []
            ),
            "scale_": (
                self.scaler.scale_.tolist() if self.scaler.scale_ is not None else []
            ),
            "var_": self.scaler.var_.tolist() if self.scaler.var_ is not None else [],
            "n_samples_seen_": n_samples_seen,
        }

        # Extract OneHotEncoder parameters
        one_hot_encoder_params = {}
        if self.one_hot_encoder is not None:
            one_hot_encoder_params = {
                "categories_": [
                    cat.tolist() for cat in self.one_hot_encoder.categories_
                ],
                "feature_names_in_": self.one_hot_encoder.feature_names_in_.tolist(),
                "n_features_in_": int(self.one_hot_encoder.n_features_in_),
            }

        # Prepare state dictionary
        state = {
            "scaler_params": scaler_params,
            "feature_columns": (
                self.feature_columns if self.feature_columns is not None else []
            ),
            "column_mappings": self.column_mappings,
            "target_encoding_mappings": self.target_encoding_mappings,
            "target_type": self.target_type,
            "target_label_encoder_classes": (
                self.target_label_encoder.classes_.tolist()
                if self.target_label_encoder is not None
                else []
            ),
            "one_hot_encoder_params": one_hot_encoder_params,
        }

        # Save to JSON
        state_file = os.path.join(self.save_dir, "preprocessor_state.json")
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, state_file=None):
        """Load preprocessing state from JSON"""
        # Use provided state_file or construct the path relative to the script's location
        if state_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            state_file = os.path.join(script_dir, "preprocessing_artifacts")

        if not os.path.exists(state_file):
            logger.error(f"No saved state found at {state_file}")
            raise FileNotFoundError(f"No saved state found at {state_file}")

        logger.info(f"state_file: {state_file}")
        with open(state_file, "r") as f:
            state = json.load(f)

        self.scaler = StandardScaler()
        scaler_params = state["scaler_params"]
        if scaler_params["mean_"]:
            self.scaler.mean_ = np.array(scaler_params["mean_"])
            self.scaler.scale_ = np.array(scaler_params["scale_"])
            self.scaler.var_ = np.array(scaler_params["var_"])
            self.scaler.n_samples_seen_ = scaler_params["n_samples_seen_"]

        self.feature_columns = state["feature_columns"]
        self.column_mappings = state["column_mappings"]
        self.target_encoding_mappings = state.get("target_encoding_mappings", {})
        self.target_type = state.get("target_type", None)
        if state.get("target_label_encoder_classes"):
            self.target_label_encoder = LabelEncoder()
            self.target_label_encoder.classes_ = np.array(
                state["target_label_encoder_classes"]
            )

        # Load OneHotEncoder state
        one_hot_encoder_params = state.get("one_hot_encoder_params", {})
        if one_hot_encoder_params:
            self.one_hot_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.int32,  # type: ignore
            )
            self.one_hot_encoder.categories_ = [
                np.array(cats) for cats in one_hot_encoder_params["categories_"]
            ]
            self.one_hot_encoder.n_features_in_ = one_hot_encoder_params[
                "n_features_in_"
            ]
            # Set drop_idx_ to None to avoid issues with partial state
            self.one_hot_encoder.drop_idx_ = None

        logger.info(f"✅ Loaded preprocessing state from {state_file}")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Column mappings: {len(self.column_mappings)}")
        logger.info(f"Target encoding mappings: {len(self.target_encoding_mappings)}")
        logger.info(f"Target type: {self.target_type}")
        logger.info(
            f"OneHotEncoder features: {len(one_hot_encoder_params.get('feature_names_in_', []))}"
        )

    def process_training_data(
        self, df, target_column, column_config, excel_filename=None
    ):
        """Convenience method for processing training data"""
        logger.info("🎯 Processing training data...")
        return self.process_and_save(
            df=df,
            target_column=target_column,
            excel_filename=excel_filename or "training_processed.xlsx",
            numerical_columns=column_config.get("numerical", []),
            low_cardinality_categorical_columns=column_config.get(
                "low_cardinality_categorical", []
            ),
            high_cardinality_categorical_columns=column_config.get(
                "high_cardinality_categorical", []
            ),
            binary_columns=column_config.get("binary", []),
            datetime_columns=column_config.get("datetime", []),
            fit=True,
        )

    def process_inference_data(self, df, excel_filename=None):
        """Convenience method for processing inference data"""
        logger.info("Processing inference data...")
        self.load_state()
        return self.process_and_save(
            df=df,
            target_column=None,
            excel_filename=excel_filename or "inference_processed.xlsx",
            fit=False,
        )
