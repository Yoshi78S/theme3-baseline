import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder

"""
[Paper]
Author: Xuewei Qiu et al.
Title: "Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation."
Conference: WSDM 2024

[Code Reference]
https://github.com/QinHsiu/ICSRec
"""

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


class _FaissKMeans:
    """CPU-only wrapper around faiss.Kmeans matching the ICSRec interface."""

    def __init__(self, num_cluster, hidden_size, seed=42, niter=20, nredo=5):
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss is required for ICSRec. pip install faiss-cpu")
        self.num_cluster = num_cluster
        self.hidden_size = hidden_size
        self.seed = seed
        self.niter = niter
        self.nredo = nredo
        self.centroids = None  # torch.Tensor [K, H]
        self.index = None

    def train(self, x_np, device):
        if x_np.shape[0] <= self.num_cluster:
            # too few samples; fall back to zero centroids
            c = np.zeros((self.num_cluster, self.hidden_size), dtype=np.float32)
        else:
            kmeans = faiss.Kmeans(
                d=self.hidden_size,
                k=self.num_cluster,
                niter=self.niter,
                nredo=self.nredo,
                seed=self.seed,
                verbose=False,
                gpu=False,
            )
            kmeans.train(x_np.astype(np.float32))
            c = kmeans.centroids
            self.index = kmeans.index
        centroids = torch.from_numpy(c).to(device)
        self.centroids = F.normalize(centroids, p=2, dim=1)

    def query(self, x_np):
        if self.index is None:
            # uninitialized: assign all to 0
            n = x_np.shape[0]
            intent_id = torch.zeros(n, dtype=torch.long, device=self.centroids.device)
            return intent_id, self.centroids[intent_id]
        _, I = self.index.search(x_np.astype(np.float32), 1)
        intent_id = torch.from_numpy(I[:, 0].astype(np.int64)).to(self.centroids.device)
        return intent_id, self.centroids[intent_id]


def _mask_correlated_samples(batch_size, device):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    for i in range(batch_size):
        mask[i, batch_size + i] = False
        mask[batch_size + i, i] = False
    return mask


def _mask_correlated_samples_fneg(intent_id):
    label = intent_id.view(1, -1).expand((2, intent_id.numel())).reshape(1, -1)
    label = label.contiguous().view(-1, 1)
    return torch.eq(label, label.t()) == 0


def _info_nce(z_i, z_j, temperature, sim, f_neg=False, intent_id=None):
    batch_size = z_i.size(0)
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    if sim == 'cos':
        s = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    else:
        s = torch.mm(z, z.t()) / temperature
    sim_i_j = torch.diag(s, batch_size)
    sim_j_i = torch.diag(s, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    if f_neg and intent_id is not None:
        mask = _mask_correlated_samples_fneg(intent_id)
        negative_samples = s.clone()
        negative_samples[mask == 0] = float("-inf")
    else:
        mask = _mask_correlated_samples(batch_size, z.device)
        negative_samples = s[mask].reshape(N, -1)

    labels = torch.zeros(N, device=z.device, dtype=torch.long)
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels


class ICSRecModel(SequentialRecModel):
    def __init__(self, args):
        super(ICSRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)

        # clustering state (populated by on_epoch_start)
        self._cluster = _FaissKMeans(
            num_cluster=args.intent_num,
            hidden_size=args.hidden_size,
            seed=args.seed,
        ) if _FAISS_AVAILABLE else None
        self._cluster_ready = False

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

    @torch.no_grad()
    def on_epoch_start(self, epoch, train_dataloader, device):
        if self.args.cl_mode not in ('cf', 'f') or self._cluster is None:
            return
        self.eval()
        feats = []
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            _, input_ids, _, _, _ = batch
            out = self.forward(input_ids)[:, -1, :]
            feats.append(out.detach().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        self._cluster.train(feats, device)
        self._cluster_ready = True
        self.train()

    def _cicl_loss(self, coarse_1, coarse_2, answers):
        # last-step representations contrasted across views
        z_i = coarse_1[:, -1, :]
        z_j = coarse_2[:, -1, :]
        logits, labels = _info_nce(
            z_i, z_j, self.args.temperature, self.args.sim,
            f_neg=self.args.f_neg, intent_id=answers,
        )
        return nn.CrossEntropyLoss()(logits, labels)

    def _ficl_loss_one(self, seq_out):
        z = seq_out[:, -1, :]
        z_np = z.detach().cpu().numpy()
        intent_id, seq2v = self._cluster.query(z_np)
        seq2v = seq2v.view(seq2v.shape[0], -1)
        logits, labels = _info_nce(
            z, seq2v, self.args.temperature, self.args.sim,
            f_neg=self.args.f_neg, intent_id=intent_id,
        )
        return nn.CrossEntropyLoss()(logits, labels)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # recommendation loss via full-softmax CE (ICSRec paper)
        seq_out_1 = self.forward(input_ids)
        logits = torch.matmul(seq_out_1[:, -1, :], self.item_embeddings.weight.t())
        rec_loss = nn.CrossEntropyLoss()(logits, answers)

        # use `same_target` as the second augmented view
        seq_out_2 = self.forward(same_target) if same_target.numel() > 0 else seq_out_1

        cicl = torch.tensor(0.0, device=input_ids.device)
        ficl = torch.tensor(0.0, device=input_ids.device)
        if self.args.cl_mode in ('c', 'cf'):
            cicl = self._cicl_loss(seq_out_1, seq_out_2, answers)
        if self.args.cl_mode in ('f', 'cf') and self._cluster_ready:
            ficl = self._ficl_loss_one(seq_out_1) + self._ficl_loss_one(seq_out_2)

        return self.args.rec_weight * rec_loss + self.args.lambda_0 * cicl + self.args.beta_0 * ficl
