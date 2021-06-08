import torch
import torch.nn as nn


class Network(nn.Module):
    # the neural network

    def __init__(self, indim: int, hiddim: int, outdim: int):
        # initilaize the neural network
        # indim: integer - size of the input dimension, hiddim: integer - size of the hidden dimension, outdim: integer - size of the output dimension

        # using the super method to initalize the super class (nn.Module)
        super(Network, self).__init__()
        # store the dimensions
        self.indim = indim
        self.hiddim = hiddim
        self.outdim = outdim
        # create the linear layers (in --> hid, hid --> out)
        self.lin1 = nn.Linear(indim, hiddim)
        self.lin2 = nn.Linear(hiddim, outdim)
        # create the sigmoid function (activation function)
        self.sig = nn.Sigmoid()


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # the forward function calculates the scores in the order: lin1 --> sig --> lin2
        # value: Tensor - to calculate the scores of
        
        # apply the first linear layer
        v1 = self.lin1(value)
        # apply the activation function
        v2 = self.sig(v1)
        # apply the second linear layer
        output = self.lin2(v2)
        # return the scores
        return output