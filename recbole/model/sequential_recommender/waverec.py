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

        # self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        )

        self.activation = nn.LeakyReLU(negative_slope=0.2)
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

        stacked = torch.stack([input_emb, low_freq_component, high_freq_component], dim=2)
        reshaped = stacked.view(-1, 50, 3, 8, 8)

        # 调整输入形状以适应卷积层
        reshaped = reshaped.permute(0, 1, 3, 4, 2).contiguous()  # Shape: (256, 50, 8, 8, 3)
        reshaped = reshaped.view(-1, 3, 8, 8)  # Shape: (256 * 50, 3, 8, 8)

        # 通过卷积层进行特征融合
        fused = self.conv(reshaped)  # Shape: (256 * 50, 1, 8, 8)

        # 恢复形状
        fused = fused.view(-1, 50, 8, 8, 1).permute(0, 1, 4, 2, 3).squeeze(2)  # Shape: (256, 50, 8, 8)
        fused = fused.view(-1,50,64)

        fused = self.LayerNorm(fused)
        # fused = self.dropout(fused)

        output = self.trm_encoder(fused, extended_attention_mask, output_all_encoded_layers=False)[0]

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # 计算相似度 查询表示和键表示进行矩阵乘法
        loss = self.loss_fct(logits, pos_items)

        return loss

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

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
