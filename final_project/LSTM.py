import torch.nn as nn


class LSTMAutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=False):

        super(LSTMAutoEncoder, self).__init__()

    def predict_Q(self, s):

        return 0


def train(model: LSTMAutoEncoder, device, transitions, targets, learning_rate):

    model = model.to(device)
    model.train()

