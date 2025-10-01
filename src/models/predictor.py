import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):
    """
    A configurable multi-layer perceptron (MLP) for binary classification with optional
    BatchNorm/Dropout and flexible activations; outputs a single logit.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        activation="relu",
        use_batch_norm=True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Activation selection
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "swish":
            act_fn = nn.SiLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Binary classification output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()  # Returns raw logit, not probability

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


class ImprovedPredictor(Predictor):
    """
    An enhanced MLP that adds input normalization, optional residual (skip) connections,
    and a regularized two-stage output head for stability and performance
    """
    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        activation="relu",
        use_batch_norm=True,
        use_residual=False,
        **kwargs,
    ):
        super().__init__(
            input_dim, hidden_dims, dropout_rate, activation, use_batch_norm
        )
        self.use_residual = use_residual
        self.input_norm = nn.BatchNorm1d(input_dim)
        # Override layers for residuals
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layer_block = nn.ModuleList()
            # Linear layer
            layer_block.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization
            if use_batch_norm:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
            # Activation
            if activation == "relu":
                layer_block.append(nn.ReLU())
            elif activation == "leaky_relu":
                layer_block.append(nn.LeakyReLU(0.01))
            elif activation == "gelu":
                layer_block.append(nn.GELU())
            elif activation == "swish":
                layer_block.append(nn.SiLU())
            # Dropout
            layer_block.append(nn.Dropout(dropout_rate))
            # Append the complete layer block once
            self.layers.append(layer_block)
            # Residual connection projection
            if use_residual and prev_dim != hidden_dim:
                setattr(self, f"residual_proj_{i}", nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer with additional regularization
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        for i, layer_block in enumerate(self.layers):
            residual = x
            for module in layer_block:
                x = module(x)
            if self.use_residual:
                if hasattr(self, f"residual_proj_{i}"):
                    residual = getattr(self, f"residual_proj_{i}")(residual)
                if x.shape == residual.shape:
                    x += residual
        return self.output_layer(x).squeeze()


class FocalLoss(nn.Module):
    """
    FocalLoss is designed to be plugged into training instead of a standard loss like nn.BCEWithLogitsLoss
    Pair any of the models with FocalLoss if the data is imbalanced and want to focus learning on
    hard-to-classify examples
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class LogisticRegression(nn.Module):
    """
    A classic logistic regression—single linear layer producing a logit; simplest,
    most interpretable baseline.
    """
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        # A pure logistic regression classifier
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Output raw logits (sigmoid applied in loss function or during inference)
        return self.linear(x).squeeze()

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "input_dim": self.input_dim,
            "hidden_dims": [],  # No hidden layers
            "dropout_rate": 0.0,  # No dropout in logistic regression
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
