import copy
import math
from typing import Generic, Iterable, Iterator, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, randint

from test_nn_template.modules.module import FeedForward, MultiHeadAttention

M = TypeVar("M", bound=torch.nn.Module)
T = TypeVar("T")


class TypedModuleList(torch.nn.ModuleList, Generic[M]):
    def __getitem__(self, idx: int) -> M:
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: M) -> None:
        return super().__setitem__(idx, module)

    def __iter__(self) -> Iterator[M]:
        return super().__iter__()

    def __iadd__(self: T, modules: Iterable[M]) -> T:
        return super().__iadd__(modules)

    def insert(self, index: int, module: M) -> None:
        super().insert(index, module)

    def append(self: T, module: M) -> T:
        return super().append(module)

    def extend(self: T, modules: Iterable[M]) -> T:
        return super().extend(modules)

    def forward(self):
        raise NotImplementedError()


class PositionalEncoding(nn.Module):
    """encodes the position along the sequence into a vector of size d_model."""

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer("positional_encodings", get_positional_encoding(d_model, max_len), False)

    def forward(self, x: Tensor):
        pe: Tensor = self.positional_encodings[: x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


def get_positional_encoding(d_model: int, max_len: int = 5000) -> Tensor:
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class EmbeddingsWithPositionalEncoding(nn.Module):
    """Embed tokens and add fixed positional encoding."""

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer("positional_encodings", get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):
    """This can act as an encoder layer or a decoder layer."""

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_prob: float,
        src_attn: Optional[MultiHeadAttention] = None,
    ):
        super().__init__()
        """Applies self-attention and ffnn to input embedding.

        If used as a decoder it applies source attention to the encoder embedding also, and adds it.
        Parameters
        ----------
        d_model : int
            Number of features in embedding
        self_attn : MultiHeadAttention
            Attention layer
        feed_forward : FeedForward
            Feed-forwards layer for the attention output
        dropout_prob : float
            Dropour prob for after self attention and ffnn
        src_attn : MultiHeadAttention, optional
            Attention layer for the encoder embedding (if this is a decoder), by default None
        """
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, src: Optional[Tensor] = None, src_mask: Optional[Tensor] = None
    ):
        z = self.norm_self_attn(x)
        z = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(z)
        if src is not None:
            z = self.norm_src_attn(x)
            z = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(x)
        z = self.norm_ff(x)
        z = self.feed_forward(z)
        x = x + self.dropout(z)
        return x


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = TypedModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Decoder(nn.Module):
    """Transformer Decoder."""

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = TypedModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Generator(nn.Module):
    """Predicts the tokens and gives the lof softmax of those."""

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)


class EncoderDecoder(nn.Module):
    """Combined Encoder-Decoder."""

    def __init__(
        self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run the source through encoder
        enc = self.encode(src, src_mask)
        # Run encodings and targets through decoder
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


if __name__ == "__main__":

    def _test_positional_encoding():
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))
        pe = get_positional_encoding(20, 100)
        plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
        plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
        plt.title("Positional encoding")
        plt.show()

    def _test_embeddings_with_positional_encoding():
        d_model = 10
        n_vocab = 20
        max_len = 3
        # batches = 2
        enc = EmbeddingsWithPositionalEncoding(d_model, n_vocab, max_len)
        x = randint(size=(max_len, d_model), high=max_len + 1)
        x_end = enc(x)
        assert x.shape == x_end[:, :, 0].shape

    def _test_transformer_layer():
        d_model = 100
        dropout_prob = 0.1
        self_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        src_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_model // 2, dropout=0.1)
        trans_layer = TransformerLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            feed_forward=feed_forward,
            self_attn=self_attn,
            src_attn=src_attn,
        )
        batch = 10
        seq = 5
        x = torch.rand((batch, seq, d_model))
        z = trans_layer(x)
        assert x.shape == z.shape
        assert not torch.equal(x, z)

    def _test_encoder():
        d_model = 100
        dropout_prob = 0.1
        self_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        src_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_model // 2, dropout=0.1)
        trans_layer = TransformerLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            feed_forward=feed_forward,
            self_attn=self_attn,
            src_attn=src_attn,
        )
        enc = Encoder(trans_layer, 5)
        batch = 10
        seq = 5
        x = torch.rand((batch, seq, d_model))
        z = enc(x)
        assert x.shape == z.shape
        assert not torch.equal(x, z)

    def _test_decoder():
        d_model = 100
        dropout_prob = 0.1
        self_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        src_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_model // 2, dropout=0.1)
        trans_layer = TransformerLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            feed_forward=feed_forward,
            self_attn=self_attn,
            src_attn=src_attn,
        )
        enc = Encoder(trans_layer, 5)
        dec = Decoder(trans_layer, 5)
        batch = 10
        seq = 5
        x = torch.rand((batch, seq, d_model))
        src = enc(x)
        z1 = dec(x)
        assert x.shape == z1.shape
        assert not torch.equal(x, z1)
        z2 = dec(x, memory=src)
        assert x.shape == z2.shape
        assert not torch.equal(x, z2)
        assert not torch.equal(z1, z2)

    def _test_encoder_decoder():
        d_model = 100
        dropout_prob = 0.1
        self_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        src_attn = MultiHeadAttention(heads=2, d_model=d_model, dropout_prob=0.1)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_model // 2, dropout=0.1)
        trans_layer = TransformerLayer(
            d_model=d_model,
            dropout_prob=dropout_prob,
            feed_forward=feed_forward,
            self_attn=self_attn,
            src_attn=src_attn,
        )
        enc = Encoder(trans_layer, 5)
        dec = Decoder(trans_layer, 5)
        batch = 10
        seq = 5
        x = torch.rand((batch, seq, d_model))
        src = enc(x)
        gen = Generator(10, d_model)
        z = EncoderDecoder(enc, dec, src, enc, enc, gen)
        assert x.shape == z.shape
        assert not torch.equal(x, z)

    # _test_positional_encoding()
    # _test_embeddings_with_positional_encoding()
    # _test_transformer_layer()
    # _test_encoder()
    # _test_decoder()
    _test_encoder_decoder
