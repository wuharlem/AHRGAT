"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import torch
from common.sublayer import PositionwiseFeedForward, MultiHeadedAttention


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RGATLayer(nn.Module):
    def __init__(
        self, d_model, heads, d_ff, dropout, att_drop=0.1, use_structure=True, dep_dim=30, alpha=1.0, beta=1.0
    ):
        super(RGATLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=att_drop, use_structure=use_structure, structure_dim=dep_dim, alpha=alpha, beta=beta
        )

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, key_padding_mask=None, structure=None):
        """
    Args:
       input (`FloatTensor`): set of `key_len`
            key vectors `[batch, seq_len, H]`
       mask: binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       key_padding_mask: binary padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
    return:
       res:  [batch, seq_len, H]
    """

        input_norm = self.layer_norm(inputs)
        context, top_attn = self.self_attn(
            input_norm,
            input_norm,
            input_norm,
            mask=mask,
            key_padding_mask=key_padding_mask,
            structure=structure,
        )
        out = self.dropout(context) + inputs
        return self.feed_forward(out), top_attn


class RGATEncoder(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        att_drop=0.1,
        use_structure=True,
        dep_dim=30,
        alpha=1.0,
        beta=1.0,
    ):
        super(RGATEncoder, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [
                RGATLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    att_drop=att_drop,
                    use_structure=use_structure,
                    dep_dim=dep_dim,
                    alpha=alpha,
                    beta=beta,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def _check_args(self, src, lengths=None):
        _, n_batch = src.size()
        if lengths is not None:
            (n_batch_,) = lengths.size()
            # aeq(n_batch, n_batch_)

    def forward(self, src, adj=None, src_key_padding_mask=None, mask=None, structure=None):
        """ See :obj:`EncoderBase.forward()`"""
        """
    Args:
       src (`FloatTensor`): set of vectors `[batch, seq_len, H]`
       mask: binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       src_key_padding_mask: binary key padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
    return:
       out_trans (`FloatTensor`): `[batch, seq_len, H]`

    """
        # self._check_args(src, lengths)

        out = src  # [B, seq_len, H]
        all_attention = []

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out, top_attn = self.transformer[i](out, mask, src_key_padding_mask, structure=structure)
            all_attention.append(top_attn)
        out = self.layer_norm(out)  # [B, seq, H]
        return out, all_attention


def sequence_mask(lengths, max_len=None):
    '''
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    '''
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(batch_size, max_len)>=(lengths.unsqueeze(1))).unsqueeze(1)
