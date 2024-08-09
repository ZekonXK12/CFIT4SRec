import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder

from mamba_ssm import Mamba


class WaveRec2(SequentialRecommender):
    def __init__(self, config, dataset):
        super(WaveRec2, self).__init__(config, dataset)

        self.n_layers = 4
        self.num_layers = 2
        self.n_heads = 4
        self.hidden_size = 64
        self.inner_size = 258
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

        self.lmd = config['lmd']

        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
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

        self.dwt = DWTForward(J=3, wave='db4', mode='zero').cuda()
        self.idwt = DWTInverse(wave='db4', mode='zero').cuda()

        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        # )

        self.upsampler = UpSampler()

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=64,
                d_state=64,
                d_conv=4,
                expand=2,
                dropout=0.3,
                num_layers=2,
            ) for _ in range(self.num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
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

        stacked = torch.stack([input_emb, up_l_emb, up_h_emb], dim=-1)
        reshaped = stacked.permute(0, 3, 1, 2).contiguous()

        # Feature fusion using conv layer
        fused = self.conv(reshaped)
        fused = fused.squeeze(1)

        fused2 = fused

        for i in range(self.num_layers):
            fused = self.mamba_layers[i](fused)
            fused = self.LayerNorm(fused)
            fused2 = self.mamba_layers[i](fused2)
            fused = self.LayerNorm(fused)

        fused = self.trm_encoder(fused, extended_attention_mask, output_all_encoded_layers=False)[0]
        fused2 = self.trm_encoder(fused2, extended_attention_mask, output_all_encoded_layers=False)[0]

        return fused, fused2

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, output2 = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        output2 = self.gather_indexes(output2, item_seq_len - 1)

        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight[:self.n_items]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        nce_loss_l = self.ncelosss(self.tau, 'cuda', seq_output, output2)

        return loss + self.lmd * nce_loss_l

    def wavelet_transform(self, input_emb):
        if input_emb.dim() != 4:
            input_emb = input_emb.unsqueeze(1)

        # Perform DWT
        Yl, Yh = self.dwt(input_emb)

        Yh_zeros = [torch.zeros_like(Yh_level) for Yh_level in Yh]
        low_freq_component = self.idwt((Yl, Yh_zeros))

        high_freq_component = input_emb - low_freq_component

        low_freq_component = low_freq_component.squeeze(1)
        high_freq_component = high_freq_component.squeeze(1)

        return low_freq_component, high_freq_component

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

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
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

        self.w_1 = nn.Linear(d_model, d_model * 4)
        self.w_2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)

        ffn_hidden_states = self.w_1(hidden_states)
        ffn_hidden_states = self.activation(ffn_hidden_states)
        ffn_hidden_states = self.ffn_dropout(ffn_hidden_states)
        ffn_hidden_states = self.w_2(ffn_hidden_states)
        ffn_hidden_states = self.ffn_dropout(ffn_hidden_states)
        hidden_states = self.ffn_LayerNorm(ffn_hidden_states + hidden_states)

        return hidden_states
