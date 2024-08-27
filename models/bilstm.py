import torch
import torch.nn as nn

class CharBiLSTM(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        hidden_dim, 
        dropout, 
        bidirect_flag = True
    ):
        super(CharBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirect_flag)

    def get_all_hiddens(self, input, char_hidden = None):

        output = self.dropout(input)
        output, char_hidden = self.lstm(output, char_hidden)
        return output.transpose(1,0), char_hidden

    def forward(self, input, char_hidden=None):
        return self.get_all_hiddens(input, char_hidden)

class BiLSTM(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dropout_ff,
    ):
        super().__init__()
        self.lstm = nn.ModuleList([])
        for _ in range(depth):
            self.lstm.append(CharBiLSTM(dim, dim, dropout_ff, bidirect_flag = True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        for lstm in self.lstm:
            x, _ = lstm(x)

        return x, x