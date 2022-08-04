import torch
import torch.nn as nn


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg  # trg = [trg len, batch size]

        optimizer.zero_grad()
        output = model(src, trg)  # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)    # output = [(trg len -1) * batch size, output dim]
        trg = trg[1:].view(-1)      # trg = [(trg len -1) * batch size]

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), clip)  # clip gradients to prevent them from exploding
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()  # turn off dropout
    epoch_loss = 0

    with torch.no_grad():  # ensure no gradients are calculated within the block, reduce memory, and higher speed.

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg  # trg = [trg len, batch size]

            # output = [trg len, batch size, output dim]
            output = model(src, trg, 0)  # no teacher forcing

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)    # output = [(trg len -1) * batch size, output dim]
            trg = trg[1:].view(-1)      # trg = [(trg len -1) * batch size]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs