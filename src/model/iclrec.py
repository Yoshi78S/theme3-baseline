import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder
from model.icsrec import _FaissKMeans, _info_nce, _FAISS_AVAILABLE

"""
[Paper]
Author: Yongjun Chen et al.
Title: "Intent Contrastive Learning for Sequential Recommendation."
Conference: WWW 2022

[Code Reference]
https://github.com/salesforce/ICLRec
"""


class ICLRecModel(SequentialRecModel):
    def __init__(self, args):
        super(ICLRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)

        self.num_intent_clusters = [int(i) for i in str(args.num_intent_clusters).split(",")]
        if _FAISS_AVAILABLE:
            if args.seq_representation_type == "mean":
                cluster_hidden = args.hidden_size
            else:
                cluster_hidden = args.hidden_size * args.max_seq_length
            self._clusters = [
                _FaissKMeans(num_cluster=k, hidden_size=cluster_hidden, seed=args.seed)
                for k in self.num_intent_clusters
            ]
        else:
            self._clusters = []
        self._clusters_ready = False

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def _seq_repr(self, seq_out):
        if self.args.seq_representation_type == "mean":
            return torch.mean(seq_out, dim=1, keepdim=False)
        return seq_out.reshape(seq_out.size(0), -1)

    @torch.no_grad()
    def on_epoch_start(self, epoch, train_dataloader, device):
        if self.args.contrast_type not in ("IntentCL", "Hybrid"):
            return
        if epoch < self.args.warm_up_epoches or not self._clusters:
            return
        self.eval()
        feats = []
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            _, input_ids, _, _, _ = batch
            out = self.forward(input_ids)
            feats.append(self._seq_repr(out).detach().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        for cluster in self._clusters:
            cluster.train(feats, device)
        self._clusters_ready = True
        self.train()

    def _instance_cl_loss(self, v1, v2, answers):
        z_i = self._seq_repr(v1)
        z_j = self._seq_repr(v2)
        logits, labels = _info_nce(
            z_i, z_j, self.args.temperature, self.args.sim,
            f_neg=self.args.de_noise, intent_id=answers,
        )
        return nn.CrossEntropyLoss()(logits, labels)

    def _intent_cl_loss(self, v1, v2):
        z_i = self._seq_repr(v1)
        z_j = self._seq_repr(v2)
        z_np = z_i.detach().cpu().numpy()
        total = 0.0
        n_used = 0
        for cluster in self._clusters:
            intent_id, seq2intent = cluster.query(z_np)
            logits_i, labels_i = _info_nce(
                z_i, seq2intent, self.args.temperature, self.args.sim,
                f_neg=self.args.de_noise, intent_id=intent_id,
            )
            logits_j, labels_j = _info_nce(
                z_j, seq2intent, self.args.temperature, self.args.sim,
                f_neg=self.args.de_noise, intent_id=intent_id,
            )
            total = total + nn.CrossEntropyLoss()(logits_i, labels_i) + nn.CrossEntropyLoss()(logits_j, labels_j)
            n_used += 2
        if n_used == 0:
            return torch.tensor(0.0, device=v1.device)
        return total / n_used

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out_1 = self.forward(input_ids)

        # full-softmax CE on last position
        logits = torch.matmul(seq_out_1[:, -1, :], self.item_embeddings.weight.t())
        rec_loss = nn.CrossEntropyLoss()(logits, answers)

        # augmented view: use same_target (falls back to dropout re-forward)
        if same_target.numel() > 0:
            seq_out_2 = self.forward(same_target)
        else:
            seq_out_2 = self.forward(input_ids)

        cl_total = torch.tensor(0.0, device=input_ids.device)
        if self.args.contrast_type in ("InstanceCL", "Hybrid"):
            cl_total = cl_total + self.args.cf_weight * self._instance_cl_loss(seq_out_1, seq_out_2, answers)
        if self.args.contrast_type in ("IntentCL", "Hybrid") and self._clusters_ready:
            cl_total = cl_total + self.args.intent_cf_weight * self._intent_cl_loss(seq_out_1, seq_out_2)

        return self.args.rec_weight * rec_loss + cl_total
