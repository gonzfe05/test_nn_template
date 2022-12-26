import math
from typing import List, Optional

from torch import Tensor, einsum, nn, rand


# Based on https://nn.labml.ai/transformers/mha.html
class PrepareForMultiHeadAttention(nn.Module):
    """Linear transformation and split a vector into heads."""

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        """Set up the head parameters

        Parameters
        ----------
        d_model : int
            Features of input vector
        heads : int
            Number of heads
        d_k : int
            Features of each head vector.
        bias : bool
            Linear transformation bias
        """
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: Tensor) -> Tensor:
        """Applies linear transformation to input vector and splits into heads."""
        # Last index is input vector features
        head_shape: int = x.shape[:-1]
        x: Tensor = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    """Computes scaled multihead attention."""

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        """Computes matching between key and query by dot-product and retrieves values conditioned on the match.

        Attention(Q,K,V) = frac{softmax(Q,K^t)}{sqrt(d_k)} * V
        Parameters
        ----------
        heads : int
            Number of heads
        d_model : int
            Dimention of input vector
        dropout_prob : float, optional
            Dropout after attention, by default 0.1
        bias : bool, optional
            For query and key, by default True
        """
        # Per-head features
        # d_model ~ heads * d_k = query.shape[-2] * query.shape[-1], same for key and value
        assert d_model % heads == 0, f"{d_model} features not divisible by the number of heads ({heads})"
        self.d_k = d_model // heads
        self.head = heads
        # dims: [batchs, seq_len, heads, features]
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)
        # dim=2 is key dimention on the QK^T matrix
        self.softmax = nn.Softmax(dim=2)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: Tensor, key: Tensor) -> Tensor:
        """Sum over query and key features, same as QK^T."""
        return einsum("bihd,bjhd->bijh", query, key)

    def prepare_mask(self, mask: Tensor, query_shape: List[int], key_shape: List[int]) -> Tensor:
        # Mask has shape [batch, seq_len_q, seq_len_k]
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == 1 or mask.shape[1] == query_shape[1]
        assert mask.shape[2] == key_shape[1]
        # Applies same mask to all heads
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Computes multi-headead attention.

        Parameters
        ----------
        query : Tensor
            Shape [batchs, seq_len, d_model]
        key : Tensor
            Shape [batchs, seq_len, d_model]
        value : Tensor
            Shape [batchs, seq_len, d_model]
        mask : Optional[Tensor], optional
            Shape [batch_size, seq_len, seq_len], by default None

        Returns
        -------
        Tensor
            Attended tensor
        """
        batch_size, seq_len, _ = query.shape
        if mask:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        # Prepate multi-heads, shape: [batches, seq, heads, d_k]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        # Compute QK^T, shape: [Batches, seq_q, seq_k, heads]
        scores = self.get_scores(query, key)
        # Scale by sqrt of features
        scores *= self.scale
        if mask:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # softmax across the key dimention
        attn = self.softmax(scores)
        # Multiply by values
        x = einsum("bijh,bjhd->bihd", attn, value)
        self.attn = attn.detach()
        # concatenate heads
        x = x.reshape(batch_size, seq_len, -1)
        return self.output(x)


if __name__ == "__main__":

    def test_attention(batches: int = 10, seq: int = 5, features: int = 20, heads: int = 4):
        d_k = features // heads
        query = rand((batches, seq, features))
        key = rand((batches, seq, features))
        value = rand((batches, seq, features))
        prep = PrepareForMultiHeadAttention(features, heads, d_k, True)
        x: Tensor = prep(query)
        assert x.shape == (batches, seq, heads, d_k)
        mha = MultiHeadAttention(heads, features, 0.1, True)
        x: Tensor = mha(query, key, value)
        assert x.shape == (batches, seq, features)
        print(f"Output: {x.shape}")

    test_attention()
