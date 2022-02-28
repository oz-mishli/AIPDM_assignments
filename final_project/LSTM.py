import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.optim as opt


class LSTMNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_actions, duelling_network, batch_first=False):

        super(LSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.duelling_net = duelling_network
        self.num_actions = num_actions

        self.lstm_layers = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=batch_first)

        if self.duelling_net:
            self.V_linear = nn.Linear(hidden_size, 1, bias=True)
            self.A_linear = nn.Linear(hidden_size, self.num_actions)
        else:
            self.Q_linear = nn.Linear(hidden_size, self.num_actions)

        #self = self.float()

    def predict_Q(self, s):

        # Convert the state(s) to a tensor in the matching dimensions before feeding to the LSTM NN
        in_tensor = torch.from_numpy(s)
        if len(in_tensor.shape) == 2:
            if self.batch_first:
                squeeze_ind = 0
            else:
                squeeze_ind = 1
            in_tensor = torch.unsqueeze(in_tensor, squeeze_ind)

        # Take the last hidden state and use it as an input to the next FC layers
        #lstm_output = self.lstm_layers(s)[:, self.hidden_size - 1, :]

        lstm_output = self.lstm_layers(in_tensor.float())[0][in_tensor.shape[0] - 1, :, :]


        if self.duelling_net:
            V = torch.nn.functional.leaky_relu(self.V_linear(lstm_output))
            A = torch.nn.functional.leaky_relu(self.A_linear(lstm_output))

            Q = V + A + torch.sum(A) / self.num_actions
        else:
            Q = torch.nn.functional.leaky_relu(self.Q_linear(lstm_output))

        return Q.detach().numpy()


def train(model: LSTMNetwork, device, transitions, targets, learning_rate):
    """
    Training the model on a single batch, as required in DQN algorithm.
    :param model: the model to be trained
    :param device: device on which the training is performed
    :param transitions: the batch of transitions data recorded during simulation
    :param targets: respective targets for the transitions (Q value for a specific action as recorded in the transition)
    :param learning_rate: learning rate hyper parameter
    :return: The loss for this training step (i.e. for the current batch)
    """

    # Switch model to training mode and initialize variables
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = opt.Adam(model.parameters(), learning_rate)

    # Load batch date to relevant device (GPU) and reset gradients
    data, target = transitions.to(device), targets.to(device)
    optimizer.zero_grad()

    # Forward-pass on the batch # TODO make sure all dimensions make sense between input (transitions) and Q output
    output_Q = model.predict_Q(data)

    # Compute loss
    loss = criterion(output_Q, target)

    loss.backward()
    optimizer.step()

    return loss



