import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):
    """Simple Feedforward Neural Network for predicting classification target"""

    def __init__(
        self, input_dim, hidden_dims=[64], dropout_rate=0.0, activation="relu"
    ):
        super(Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            logger.error(f"Unsupported activation: {activation}")
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (single neuron for classification)
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
