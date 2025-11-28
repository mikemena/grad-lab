# Model Training Configuration Reference

This document provides a comprehensive guide to configuring model training experiments. The YAML configuration files control model architecture, training parameters, hyperparameter tuning, preprocessing, inference settings, and data paths.

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Model Architecture](#model-architecture)
3. [Training Parameters](#training-parameters)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Preprocessing](#preprocessing)
6. [Inference Settings](#inference-settings)
7. [Data Configuration](#data-configuration)
8. [Best Practices & Tips](#best-practices--tips)
9. [Example Configurations](#example-configurations)

---

## Configuration Overview

Each configuration file is organized into six main sections:

| Section | Purpose |
|---------|---------|
| `model` | Defines the model architecture and type |
| `training` | Controls the training loop and optimization |
| `tuning` | Configures hyperparameter search |
| `preprocessing` | Sets data preprocessing options |
| `inference` | Defines cost-sensitive decision thresholds |
| `data` | Specifies file paths and target column |

---

## Model Architecture

### Model Types

```yaml
model:
  type: <model_type>
```

| Type | Description | Use Case |
|------|-------------|----------|
| `basic` | Simple neural network (Predictor) | Quick baseline, small datasets |
| `improved` | Enhanced neural network (ImprovedPredictor) | Better regularization, complex patterns |
| `logistic` | Logistic regression | Interpretable baseline, linear relationships |
| `rf` | Random Forest | Robust ensemble, handles non-linear relationships |
| `xgb` | XGBoost | State-of-the-art gradient boosting |

### Neural Network Options

These parameters apply to `basic` and `improved` model types:

```yaml
model:
  input_dim: 14          # Number of input features (use 'auto' for dynamic detection)
  hidden_dims: [256, 128, 64]  # Hidden layer sizes
  dropout_rate: 0.3      # Dropout probability (0.0 - 1.0)
  activation: relu       # Activation function
  use_batch_norm: true   # Enable batch normalization
  use_residual: false    # Enable residual/skip connections
```

**Activation Functions:**

| Activation | Description | When to Use |
|------------|-------------|-------------|
| `relu` | Rectified Linear Unit | Default choice, fast training |
| `gelu` | Gaussian Error Linear Unit | Transformer-style networks |
| `leaky_relu` | Leaky ReLU | Prevents dying neurons |
| `swish` | Self-gated activation | Smooth gradients, deeper networks |

**Guidelines for `hidden_dims`:**
- Start shallow: `[64]` or `[128]`
- Add depth for complex data: `[256, 128, 64]`
- Typically decrease layer size as you go deeper
- More layers ≠ better performance (risk of overfitting)

### Tree-Based Model Options

Shared parameters for `rf` and `xgb`:

```yaml
model:
  tree_params:
    random_state: 42     # Reproducibility seed
    n_jobs: -1           # Parallel jobs (-1 = use all cores)
```

#### Random Forest Parameters

```yaml
model:
  rf_params:
    n_estimators: 400        # Number of trees
    max_depth: 10            # Maximum tree depth (null = unlimited)
    min_samples_split: 2     # Min samples to split a node
    min_samples_leaf: 1      # Min samples in a leaf
    class_weight: balanced   # Handle class imbalance
    criterion: gini          # Split criterion: 'gini' or 'entropy'
```

#### XGBoost Parameters

```yaml
model:
  xgb_params:
    n_estimators: 600        # Number of boosting rounds
    learning_rate: 0.05      # Step size shrinkage
    max_depth: 6             # Maximum tree depth
    subsample: 0.8           # Row subsampling ratio
    colsample_bytree: 0.8    # Column subsampling ratio
    reg_lambda: 1.0          # L2 regularization
    scale_pos_weight: null   # Balance positive/negative weights (auto-fill at runtime)
    tree_method: "hist"      # Tree construction algorithm
    eval_metric: "logloss"   # Evaluation metric
```

---

## Training Parameters

```yaml
training:
  run_name: tokyo            # Experiment identifier
  epochs: 50                 # Maximum training epochs
  lr: 0.001                  # Learning rate
  loss_type: bce             # Loss function
  alpha: 0.25                # Focal/weighted loss parameter
  gamma: 2.0                 # Focal loss focusing parameter
  optimizer_name: AdamW      # Optimizer
  weight_decay: 0.001        # L2 regularization strength
  patience: 10               # Early stopping patience
  min_delta: 0.0001          # Minimum improvement threshold
  use_scaling: false         # Apply feature scaling
  use_class_weights: true    # Weight classes by frequency
  use_scheduler: false       # Enable learning rate scheduler
  scheduler_type: cosine     # Scheduler type (if enabled)
  batch_size: 32             # Mini-batch size
```

### Loss Functions

| Loss Type | Description | When to Use |
|-----------|-------------|-------------|
| `bce` | Binary Cross-Entropy | Balanced datasets, standard classification |
| `weighted_bce` | Weighted BCE | Moderate class imbalance |
| `focal` | Focal Loss | Severe class imbalance, hard examples |

**Focal Loss Parameters:**
- `alpha`: Weight for the positive class (typically 0.25)
- `gamma`: Focusing parameter; higher values down-weight easy examples (typically 2.0)

### Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `Adam` | Adaptive moment estimation | General purpose, default choice |
| `AdamW` | Adam with decoupled weight decay | Better generalization |

### Learning Rate Schedulers

When `use_scheduler: true`:

| Scheduler | Behavior |
|-----------|----------|
| `cosine` | Smooth cosine annealing to near-zero |
| (others can be added) | — |

---

## Hyperparameter Tuning

Enable grid search for automated hyperparameter optimization:

```yaml
tuning:
  enabled: true              # Toggle tuning on/off
  calibrate: true            # Apply probability calibration
  
  # Neural network hyperparameters
  lr_range: [0.0005, 0.001, 0.005, 0.01]
  hidden_dims_options: [[128], [256, 128], [512, 256, 128]]
  dropout_range: [0.2, 0.3, 0.4, 0.5]
  
  # Tree-based hyperparameters
  n_estimators_range: [200, 400, 600]
  max_depth_range: [4, 6, 8, 10]
  min_samples_split_range: [2, 5]
  min_samples_leaf_range: [1, 2]
  subsample_range: [0.7, 0.8, 0.9]
  colsample_bytree: 0.8
  criterion: "gini"
```

**Important:** Set `enabled: false` when running a single experiment with fixed parameters.

---

## Preprocessing

```yaml
preprocessing:
  save_dir: experiments/preprocessing/artifacts
  apply_scaling: true                    # StandardScaler / normalization
  enable_feature_engineering: true       # Auto-binning and feature creation
```

| Option | Neural Networks | Tree Models |
|--------|-----------------|-------------|
| `apply_scaling` | **Recommended: true** | **Recommended: false** |
| `enable_feature_engineering` | Optional | Optional |

> **Tip:** Tree-based models (RF, XGBoost) are scale-invariant; scaling adds overhead without benefit.

---

## Inference Settings

Configure cost-sensitive decision making:

```yaml
inference:
  cost_false_positives: 10     # Cost of false positive
  cost_false_negatives: 25     # Cost of false negative
  benefit_true_positives: 20   # Benefit of true positive
  decision_threshold: 0.39     # Classification threshold
```

### Cost-Sensitive Threshold Optimization

The `decision_threshold` shifts the classification boundary based on business costs:

- **Lower threshold** (e.g., 0.26): More positive predictions → fewer false negatives
- **Higher threshold** (e.g., 0.50): Fewer positive predictions → fewer false positives

**Optimal threshold calculation:**

```
threshold = cost_FP / (cost_FP + cost_FN)
```

For the default costs (FP=10, FN=25):
```
threshold = 10 / (10 + 25) = 0.286
```

---

## Data Configuration

```yaml
data:
  filepath:
    state: experiments/preprocessing/artifacts/preprocessor_state.json
    train: experiments/preprocessing/artifacts/titanic_train_processed.xlsx
    val: experiments/preprocessing/artifacts/titanic_val_processed.xlsx
    test: experiments/preprocessing/artifacts/titanic_test_processed.xlsx
  target_column: Survived
```

| Field | Description |
|-------|-------------|
| `state` | Preprocessor state for consistent transforms |
| `train` | Training data file |
| `val` | Validation data file |
| `test` | Test/holdout data file |
| `target_column` | Name of the label column |

---

## Best Practices & Tips

### General Recommendations

1. **Start simple**: Begin with `logistic` or `basic` to establish a baseline before trying complex models.

2. **Version your configs**: Use meaningful `run_name` values (e.g., `tokyo`, `miami`, `sydney`) to track experiments.

3. **Match scaling to model type**:
   - Neural networks: `apply_scaling: true`
   - Tree models: `apply_scaling: false`

4. **Use early stopping**: Set `patience` to prevent overfitting while allowing sufficient training time.

### Neural Network Tips

| Symptom | Solution |
|---------|----------|
| Overfitting | Increase `dropout_rate`, add `weight_decay`, reduce `hidden_dims` |
| Underfitting | Increase `hidden_dims`, reduce `dropout_rate`, increase `epochs` |
| Unstable training | Lower `lr`, enable `use_batch_norm`, use `AdamW` |
| Slow convergence | Increase `lr`, enable `use_scheduler` |

**Recommended starting configuration:**
```yaml
model:
  type: improved
  hidden_dims: [128, 64]
  dropout_rate: 0.3
  use_batch_norm: true
training:
  lr: 0.001
  optimizer_name: AdamW
  patience: 10
```

### Tree Model Tips

| Symptom | Solution |
|---------|----------|
| Overfitting | Reduce `max_depth`, increase `min_samples_leaf`, reduce `n_estimators` |
| Underfitting | Increase `max_depth`, increase `n_estimators` |
| Slow training | Reduce `n_estimators`, use `tree_method: "hist"` for XGBoost |

**Random Forest vs XGBoost:**

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| Training speed | Faster | Slower |
| Overfitting risk | Lower | Higher (needs tuning) |
| Performance ceiling | Good | Often better |
| Interpretability | Higher | Lower |

### Class Imbalance Handling

Choose **one** approach (not multiple):

| Method | Configuration |
|--------|---------------|
| Weighted loss | `use_class_weights: true` |
| Focal loss | `loss_type: focal` with `alpha`, `gamma` |
| RF class weights | `class_weight: balanced` |
| XGBoost scaling | `scale_pos_weight: <ratio>` |

### Hyperparameter Tuning Strategy

1. **Coarse search first**: Use wide ranges with few values
2. **Refine around best**: Narrow the range around top performers
3. **Always validate**: Use the validation set, not training metrics

---

## Example Configurations

### Quick Baseline (Logistic Regression)

```yaml
model:
  type: logistic
  input_dim: 28
training:
  run_name: baseline
  epochs: 50
  lr: 0.0001
  loss_type: bce
  use_class_weights: false
preprocessing:
  apply_scaling: true
```

### Production Neural Network

```yaml
model:
  type: improved
  input_dim: auto
  hidden_dims: [256, 128, 64]
  dropout_rate: 0.3
  activation: relu
  use_batch_norm: true
training:
  run_name: production_nn
  epochs: 50
  lr: 0.001
  loss_type: bce
  optimizer_name: AdamW
  weight_decay: 0.001
  patience: 10
  use_class_weights: true
  batch_size: 32
preprocessing:
  apply_scaling: true
```

### Robust Tree Model (Random Forest)

```yaml
model:
  type: rf
  tree_params:
    random_state: 42
    n_jobs: -1
  rf_params:
    n_estimators: 400
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    class_weight: balanced
    criterion: gini
training:
  run_name: robust_rf
preprocessing:
  apply_scaling: false
```

### High-Performance XGBoost with Tuning

```yaml
model:
  type: xgb
  tree_params:
    random_state: 42
    n_jobs: -1
  xgb_params:
    n_estimators: 600
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    reg_lambda: 1.0
    tree_method: hist
    eval_metric: logloss
training:
  run_name: tuned_xgb
tuning:
  enabled: true
  n_estimators_range: [200, 400, 600]
  max_depth_range: [4, 6, 8, 10]
  subsample_range: [0.7, 0.8, 0.9]
  calibrate: true
preprocessing:
  apply_scaling: false
```

---

## Quick Reference Card

| Model Type | Scaling | Key Params | Imbalance Handling |
|------------|---------|------------|-------------------|
| `basic` | ✅ Yes | `hidden_dims`, `dropout_rate`, `lr` | `use_class_weights` |
| `improved` | ✅ Yes | `hidden_dims`, `dropout_rate`, `lr`, `use_batch_norm` | `use_class_weights`, `focal` loss |
| `logistic` | ✅ Yes | `lr` | `use_class_weights` |
| `rf` | ❌ No | `n_estimators`, `max_depth`, `min_samples_*` | `class_weight: balanced` |
| `xgb` | ❌ No | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | `scale_pos_weight` |

---
