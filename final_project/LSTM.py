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
        self.eval()

        self.lstm_layers = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=batch_first)

        if self.duelling_net:
            self.V_linear = nn.Linear(hidden_size, 1, bias=True)
            self.A_linear = nn.Linear(hidden_size, self.num_actions)
        else:
            self.Q_linear = nn.Linear(hidden_size, self.num_actions)

    def forward(self, s):
        """
        Forward pass of the DQN, returning a vector of Q values with a scalar entry for each action
        :param s: A batch of states as tensor in shape (series length, batch size, num features)
        :return: a numpy array of Q values with a scalar entry for each action
        """

        # Take the last hidden state and use it as an input to the next FC layers
        lstm_output = self.lstm_layers(s)[0][s.shape[0] - 1, :, :]

        # Execute the inference according to whether we use duelling network mode or not
        if self.duelling_net:
            V = torch.nn.functional.leaky_relu(self.V_linear(lstm_output))
            A = torch.nn.functional.leaky_relu(self.A_linear(lstm_output))

            Q = V + A + torch.sum(A) / self.num_actions
        else:
            Q = torch.nn.functional.leaky_relu(self.Q_linear(lstm_output))

        # Different behavior between train and test here
        if self.training:
            return Q
        else:
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

    # Format batch and transitions data, then load to relevant device (GPU) and reset gradients
    data = torch.swapaxes(torch.tensor(np.array([i[0] for i in transitions])), 0, 1).to(device)
    actions = torch.tensor(np.array([j[1] for j in transitions])).to(device) + 1
    target = torch.tensor(targets).to(device).float()
    optimizer.zero_grad()

    # Forward-pass on the batch
    output_Q_vals = model.forward(data.float())
    output_Q = torch.take(output_Q_vals, actions)

    # Compute loss
    loss = criterion(output_Q, target)

    loss.backward()
    optimizer.step()

    model.eval()

    return loss.item()



