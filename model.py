import random
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __int__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().Encoder.__int__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # src = [src len, batch size]

        outputs, (hidden, cell) = self.rnn(embedded)  # embedded = [src len, batch size, emb dim]

        # out = [src len, batch size, hid dim* n directions]
        return hidden, cell  # hidden, cell = [n layers* n directions, batch size, hid dim]


class Decoder(nn.Module):
    def __int__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        Input = input.unsqueeze(0)  # input = [batch size] to [1, batch size]
        embedded = self.dropout(self.embedding(Input))  # [1, batch size, emb dim]

        # hidden, cell will be [n layers, batch size, hid dim] in decoder
        output, (hidden, cell) = self.rnn(embedded,
                                          (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]

        prediction = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __int__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimension of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal numbers of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[0, :]       # first input is <sos>

        for i in range(1, trg_len):     # first element in out is 0 comparing to <sos>, rests stay the same
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[i] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[i] if teacher_force else top1

        return outputs
