import torch
import torch.nn as nn
import torch.nn.functional as F


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask > 0, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(torch.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, connect_dropout=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2),
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size * 4 + self.n_chunk * 2, dropout=connect_dropout)

        self.drop_weight_modules = [self.hh]

    def forward(self, hidden,
                transformed_input):
        hx, cx = hidden

        gates = transformed_input + self.hh(hx)
        c_in_gate, c_forget_gate = gates[:, :self.n_chunk * 2].chunk(2, 1)
        out_gate, cell, in_gate, forget_gate = gates[:, self.n_chunk * 2:].view(-1, self.n_chunk * 4,
                                                                                self.chunk_size).chunk(4, 1)

        c_in_gate = torch.ones_like(c_in_gate) - cumsoftmax(c_in_gate)
        c_forget_gate = cumsoftmax(c_forget_gate)

        c_in_gate = c_in_gate[:, :, None]
        c_forget_gate = c_forget_gate[:, :, None]

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell = torch.tanh(cell)
        out_gate = torch.sigmoid(out_gate)

        overlap = c_forget_gate * c_in_gate
        forget_gate = forget_gate * overlap + (c_forget_gate - overlap)
        in_gate = in_gate * overlap + (c_in_gate - overlap)
        cy = forget_gate * cx + in_gate * cell

        hy = out_gate * torch.tanh(cy)
        return hy.reshape(-1, self.hidden_size), cy

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, dropout, biderectional):
        super(ONLSTMStack, self).__init__()

        locked_dropout = dropout
        connect_dropout = dropout
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i + 1],
                                               chunk_size,
                                               connect_dropout=connect_dropout)
                                    for i in range(len(layer_sizes) - 1)])
        self.locked_dropout = LockedDropout()
        self.dropout = locked_dropout
        self.sizes = layer_sizes
        self.bidirectional = biderectional

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def _forward(self, x, reverse):
        length, batch_size, _ = x.size()
        hidden = self.init_hidden(x.size(1))

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = x
        output = None

        for l in range(len(self.cells)):
            curr_layer = torch.zeros_like(x)
            t_input = self.cells[l].ih(prev_layer)
            for t in range(length):
                __t = length - t - 1 if reverse else t
                hidden, cell = self.cells[l](prev_state[l], t_input[__t])
                prev_state[l] = hidden, cell
                curr_layer[t] = hidden
            prev_layer = curr_layer
            output = prev_layer
            prev_layer = self.locked_dropout(prev_layer, self.dropout)

        return output

    def forward(self, x):
        if self.bidirectional:
            return torch.cat([
                self._forward(x, reverse=False),
                self._forward(x, reverse=True)
            ], dim=-1)
        return self._forward(x, reverse=False)



class OrderedNeuronLayer(nn.Module):
    def __init__(self, args):
        super(OrderedNeuronLayer, self).__init__()
        self.embed_dim = args.encoder_embed_dim
        self.on_lstm = ONLSTMStack(
            layer_sizes=[self.embed_dim for _ in range(2)],
            chunk_size=args.chunk_size,
            dropout=args.dropout,
            biderectional=args.encoder_bidirectional,
        )
        self.bidirectional = args.encoder_bidirectional
        if self.bidirectional:
            self.fc0 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.activation_fn = nn.functional.linear

    def forward(self, x):
        x = self.on_lstm(x)
        if self.bidirectional:
            x = self.activation_fn(self.fc0(x))
        return x

#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention


class OrderedTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = OrderedNeuronLayer(args)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x = self.self_attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
