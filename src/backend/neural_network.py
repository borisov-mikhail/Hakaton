import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        result = self.layer_1(x)
        result = self.relu(result)
        result = self.layer_2(result)
        result = self.relu(result)
        result = self.layer_3(result)
        return result
