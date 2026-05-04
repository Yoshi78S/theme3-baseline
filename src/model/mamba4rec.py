import torch
import torch.nn as nn

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm

"""
[Paper]
Author: Chengkai Liu et al.
Title: "Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models."
arXiv 2403.03900 (2024)

[Code Reference]
https://github.com/chengkai-liu/Mamba4Rec
"""

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERR = None
except Exception as _e:
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERR = _e


class MambaFeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        h = self.w_1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.w_2(h)
        h = self.dropout(h)
        return self.LayerNorm(h + x)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = MambaFeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, x):
        h = self.mamba(x)
        if self.num_layers == 1:
            h = self.LayerNorm(self.dropout(h))
        else:
            h = self.LayerNorm(self.dropout(h) + x)
        return self.ffn(h)


class Mamba4RecModel(SequentialRecModel):
    def __init__(self, args):
        super(Mamba4RecModel, self).__init__(args)
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed or failed to import. "
                f"Original error: {_MAMBA_IMPORT_ERR}. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        self.args = args
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=args.hidden_size,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
                dropout=args.hidden_dropout_prob,
                num_layers=args.num_hidden_layers,
            )
            for _ in range(args.num_hidden_layers)
        ])

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        x = self.item_embeddings(input_ids)
        x = self.dropout(x)
        x = self.LayerNorm(x)

        outputs = []
        for block in self.mamba_blocks:
            x = block(x)
            outputs.append(x)

        if all_sequence_output:
            return outputs
        return outputs[-1]

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
