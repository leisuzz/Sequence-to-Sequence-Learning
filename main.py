import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import Encoder, Decoder, Seq2Seq
from train import train, evaluate, epoch_time

import spacy
import numpy as np

import random
import math
import time

SEED = 1111  # random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tockenizer(text)][::-1]  # Reverse the German text for source


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tockenizer(text)]  # English text for target


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',  # start of sentence
            eos_token='<eos>',  # end of sentence
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), field=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)  # Tokens that appear only once are <unk>
TRG.build_vocb(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      device=device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):  # initialize weights of the model
    for name, param in m.named_parmeters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_paramerters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # calculate number of trainable parameters


model.apply(init_weights)
print(f'The model has {count_paramerters(model):,} trainable parameterts')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]    # calculate loss by passing idx of <pad> token/ padding token
criterion = nn.CrossEntropyLoss(ignore_index= TRG_PAD_IDX)  # calculate avg loss per token

EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 's2s-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('s2s-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL {math.exp(test_loss):7.3f} |')



