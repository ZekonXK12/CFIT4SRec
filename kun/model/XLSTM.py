import torch
import torch.nn as nn


class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义输入门、遗忘门、输出门和候选记忆的权重
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        i_t = self.sigmoid(self.W_i(combined))
        f_t = self.sigmoid(self.W_f(combined))
        o_t = self.sigmoid(self.W_o(combined))
        c_tilda = self.tanh(self.W_c(combined))

        c_t = f_t * c_prev + i_t * c_tilda
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t


class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList(
            [xLSTMCell(input_size, hidden_size) if i == 0 else xLSTMCell(hidden_size, hidden_size) for i in
             range(num_layers)])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h_0 = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_0 = torch.zeros(batch_size, self.hidden_size).to(x.device)
            hidden = [(h_0, c_0) for _ in range(self.num_layers)]

        h_n, c_n = [], []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t, c_t = self.lstm_cells[layer](x_t, hidden[layer])
                x_t = h_t
                hidden[layer] = (h_t, c_t)
            h_n.append(h_t)
            c_n.append(c_t)

        h_n = torch.stack(h_n, dim=1)
        c_n = torch.stack(c_n, dim=1)

        return h_n, c_n
