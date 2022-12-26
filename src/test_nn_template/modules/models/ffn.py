import torch
from torch import nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        """Init params.

        Parameters
        ----------
        d_model : int
            Number of features in the input embedding
        d_ff : int
            Number of features in the hidden layer of the FFN
        dropout :  float
            Dropout probability for the hidden layer
        activation :  nn.Module
            Activation function. Default: ReLu
        is_gated : bool
            Make hidden layer gated. Default: False
        bias1 : bool
            First fully connected layer with learnable bias. Default: True
        bias2 : bool
            Second fully connected layer with learnable bias. Default: True
        bias_gate : bool
            Make the fully connected layer for the gate should with learnable bias. Default: True
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        x = self.activation(self.layer1(x))
        if self.is_gated:
            x *= self.linear_v(x)
        # Apply dropout
        x = self.dropout(x)
        return self.layer2(x)


class MinimalFeedForward(nn.Module):
    def __init__(self, num_classes: int):
        super(MinimalFeedForward, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(2, 2),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(2, num_classes)
        )

    def forward(self, x):
        output = self.model(x)
        return output
