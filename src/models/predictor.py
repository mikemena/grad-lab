import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import setup_logger  # Assuming this is in src/logger.py

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):
    """Basic Feedforward Neural Network (from v1-v3)."""

    def __init__(
        self,
        input_dim,
        hidden_dims=[64],
        dropout_rate=0.0,
        activation="relu",
        use_batch_norm=False,
    ):
        super(Predictor, self).__init__()
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
        return self.network(x).squeeze()

    # get_model_info() as in your code...


class ImprovedPredictor(Predictor):
    """Enhanced version with residuals and input norm (from v4)."""

    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.4,
        activation="swish",
        use_batch_norm=True,
        use_residual=True,
    ):
        super().__init__(
            input_dim, hidden_dims, dropout_rate, activation, use_batch_norm
        )
        self.use_residual = use_residual
        self.input_norm = nn.BatchNorm1d(input_dim)
        # Override layers for residuals (adapt from your v4 code)
        self.layers = nn.ModuleList()  # Build with residuals as in v4
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layer_block = nn.ModuleList([nn.Linear(prev_dim, hidden_dim)])
            if use_batch_norm:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
            # Add activation...
            self.layers.append(layer_block)
            if use_residual and prev_dim != hidden_dim:
                setattr(self, f"residual_proj_{i}", nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        # Output layer as in v4...

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
        return super().forward(x)  # Or custom output


class FocalLoss(nn.Module):
    """From v4."""

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
        return focal_loss
