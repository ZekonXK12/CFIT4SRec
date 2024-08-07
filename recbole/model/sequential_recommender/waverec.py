import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class WaveRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(WaveRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = 4
        self.n_heads = 4
        self.hidden_size = 64  # same as embedding_size
        self.inner_size = 258  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.initializer_range = 0.02
        self.loss_type = 'CE'

        self.shuffle_aug = True
        self.wavelet_aug = True
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

        # Initialize DWT and IDWT
        self.dwt = DWTForward(J=3, wave='db4', mode='zero').cuda()  # Single level DWT
        self.idwt = DWTInverse(wave='db4', mode='zero').cuda()

        self.dwt2 = DWTForward(J=5, wave='db4', mode='zero').cuda()
        self.idwt2 = DWTInverse(wave='db4', mode='zero').cuda()

        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        self.upsampler = UpSampler()

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

    def forward(self, item_seq, item_seq_len):
        extended_attention_mask = self.get_attention_mask(item_seq)

        input_emb = self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        low_freq_component, high_freq_component = self.wavelet_transform(input_emb)

        up_l_emb = self.upsampler(low_freq_component)
        up_h_emb = self.upsampler(high_freq_component)

        stacked = torch.stack([input_emb, up_l_emb, up_h_emb],
                              dim=-1)  # Shape: (batch_size, seq_length, hidden_size, 3)
        reshaped = stacked.permute(0, 3, 1, 2).contiguous()  # Shape: (batch_size, 3, seq_length, hidden_size)

        # Feature fusion using conv layer
        fused = self.conv(reshaped)
        fused = fused.squeeze(1)

        fused = self.LayerNorm(fused)
        output = self.trm_encoder(fused, extended_attention_mask, output_all_encoded_layers=False)[0]

        # low_freq_component2, high_freq_component2 = self.wavelet_transform2(input_emb)
        #
        # up_l_emb2 = self.upsampler(low_freq_component2)
        # up_h_emb2 = self.upsampler(high_freq_component2)
        #
        # stacked2 = torch.stack([input_emb, up_l_emb2, up_h_emb2],
        #                       dim=-1)  # Shape: (batch_size, seq_length, hidden_size, 3)
        # reshaped2 = stacked2.permute(0, 3, 1, 2).contiguous()  # Shape: (batch_size, 3, seq_length, hidden_size)
        #
        # # Feature fusion using conv layer
        # fused2 = self.conv(reshaped2)
        # fused2 = fused2.squeeze(1)
        #
        # fused2 = self.LayerNorm(fused2)
        #
        #
        # output2 = self.trm_encoder(fused2, extended_attention_mask, output_all_encoded_layers=False)[0]


        return output,input_emb,fused

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,input_emb,fused = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # 计算相似度 查询表示和键表示进行矩阵乘法
        loss = self.loss_fct(logits, pos_items)

        nce_loss= self.ncelosss(1, 'cuda', input_emb, fused)

        return loss+0.1*nce_loss

    def wavelet_transform(self, input_emb):
        # Ensure the input tensor has the correct shape (N, C, H, W)
        if input_emb.dim() != 4:
            input_emb = input_emb.unsqueeze(1)  # Add channel dimension: (N, 1, H, W)

        # Perform DWT
        Yl, Yh = self.dwt(input_emb)

        # Only perform inverse DWT on Yl
        Yh_zeros = [torch.zeros_like(Yh_level) for Yh_level in Yh]  # Create zeroed high-frequency components
        low_freq_component = self.idwt((Yl, Yh_zeros))

        # Calculate high frequency component as residual
        high_freq_component = input_emb - low_freq_component

        # Ensure the output has the same shape as input_emb
        low_freq_component = low_freq_component.squeeze(1)  # Remove channel dimension if needed
        high_freq_component = high_freq_component.squeeze(1)  # Remove channel dimension if needed

        return low_freq_component, high_freq_component

    def wavelet_transform2(self, input_emb):
        # Ensure the input tensor has the correct shape (N, C, H, W)
        if input_emb.dim() != 4:
            input_emb = input_emb.unsqueeze(1)  # Add channel dimension: (N, 1, H, W)

        # Perform DWT
        Yl, Yh = self.dwt2(input_emb)

        # Only perform inverse DWT on Yl
        Yh_zeros = [torch.zeros_like(Yh_level) for Yh_level in Yh]  # Create zeroed high-frequency components
        low_freq_component = self.idwt2((Yl, Yh_zeros))

        # Calculate high frequency component as residual
        high_freq_component = input_emb - low_freq_component

        # Ensure the output has the same shape as input_emb
        low_freq_component = low_freq_component.squeeze(1)  # Remove channel dimension if needed
        high_freq_component = high_freq_component.squeeze(1)  # Remove channel dimension if needed

        return low_freq_component, high_freq_component

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
        seq_output,_,_ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,_,_ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class UpSampler(nn.Module):
    def __init__(self):
        super(UpSampler, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 输入形状: [batch_size, seq_length, embedding_dim]
        # 需要转置为 [batch_size, embedding_dim, seq_length]
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # 转置为 [batch_size, seq_length, new_channels] 以适应 LSTM
        x, _ = self.bilstm(x)
        x = x.transpose(1, 2)  # 转置回 [batch_size, new_channels, seq_length] 以适应解码器
        x = self.decoder(x)
        x = x.transpose(1, 2)  # 转置回 [batch_size, seq_length, embedding_dim]
        return x

import torch
import torch.nn as nn

class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM 的标准门：输入门、遗忘门、输出门
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # 扩展的门：重置门（假设为新的机制）
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # 单元态更新
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hx):
        h_prev, c_prev = hx

        # 拼接输入和隐藏状态
        combined = torch.cat((input, h_prev), 1)

        # 计算门值
        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        r_t = torch.sigmoid(self.reset_gate(combined))  # 新的重置门

        # 计算候选单元态
        c_tilde = torch.tanh(self.cell_gate(combined))

        # 更新单元态和隐藏态
        c_t = f_t * c_prev + i_t * c_tilde
        # 使用重置门选择性重置某些单元
        c_t = r_t * c_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 使用多个 xLSTM 单元堆叠
        self.cells = nn.ModuleList([xLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, input):
        batch_size, seq_len, _ = input.size()
        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(input.device),
                    torch.zeros(batch_size, self.hidden_size).to(input.device))

        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]
            for layer in range(self.num_layers):
                h_t, c_t = self.cells[layer](x_t, (h_t, c_t))
                x_t = h_t  # 输出作为下一层的输入
            outputs.append(h_t.unsqueeze(1))

        # 拼接输出为整个序列
        outputs = torch.cat(outputs, dim=1)
        return outputs


