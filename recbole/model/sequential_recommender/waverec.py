import random

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.fft as fft
import torch.nn.functional as F


class WaveRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(WaveRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = 2
        self.n_heads = 2
        self.hidden_size = 64  # same as embedding_size
        self.inner_size = 258 # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = 0.5
        self.attn_dropout_prob = 0.5
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.initializer_range = 0.02
        self.loss_type = 'CE'

        self.shuffle_aug=True
        self.lmd = config['lmd']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )


        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()


        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    # def forward(self, item_seq, item_seq_len):
    #     extended_attention_mask = self.get_attention_mask(item_seq)
    #     position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
    #     position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
    #     position_embedding = self.position_embedding(position_ids)
    #     shuffled_item_seq=self.shuffle_seq(item_seq,item_seq_len, 0.5)
    #
    #
    #     item_emb = self.item_embedding(item_seq)
    #
    #     input_emb = item_emb + position_embedding
    #     input_emb = self.LayerNorm(input_emb)
    #
    #
    #     input_emb = self.dropout(input_emb)
    #
    #     input_emb= input_emb
    #     trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
    #     output = trm_output[-1]
    #
    #
    #     return output

    def forward(self, item_seq, item_seq_len):
        extended_attention_mask = self.get_attention_mask(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding



        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        if self.shuffle_aug:
            shuffled_item_seq = self.shuffle_seq(item_seq, item_seq_len, 0.5)
            shuffled_item_seq = self.item_embedding(shuffled_item_seq)
            shuffled_item_seq = shuffled_item_seq + position_embedding
            shuffled_item_seq = self.LayerNorm(shuffled_item_seq)
            shuffled_item_seq = self.dropout(shuffled_item_seq)
        trm_output_shuffled = self.trm_encoder(shuffled_item_seq,extended_attention_mask,output_all_encoded_layers=True)
        shuffled_aug_output=trm_output_shuffled[-1]

        return output,shuffled_aug_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, shuffled_aug_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        shuffled_aug_output=self.gather_indexes(shuffled_aug_output,item_seq_len-1)
        pos_items = interaction[self.POS_ITEM_ID]


        test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))    # 计算相似度 查询表示和键表示进行矩阵乘法
        loss = self.loss_fct(logits, pos_items)

        if self.shuffle_aug:
            shuffle_aug_loss=self.ncelosss(self.tau, 'cuda', seq_output,shuffled_aug_output)

        return loss+self.lmd*shuffle_aug_loss

    def ncelosss(self, temperature, device, batch_sample_one, batch_sample_two):
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        b_size = batch_sample_one.shape[0]
        batch_sample_one = batch_sample_one.view(b_size, -1)
        batch_sample_two = batch_sample_two.view(b_size, -1)

        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output= self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output= self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    # contrastive learning methods
    def shuffle_seq(self, batch_data, seq_len, shuffle_prob=0.5):
        batch_data=batch_data.clone()

        for i in range(self.batch_size):
            user_len=int(seq_len[i])
            indices=random.sample(range(user_len),int(user_len*shuffle_prob))
            indices_shuffled=indices.copy()
            random.shuffle(indices_shuffled)
            batch_data[i,indices]=batch_data[i,indices_shuffled]

        return batch_data