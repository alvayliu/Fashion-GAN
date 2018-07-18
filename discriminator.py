"""
This class defines the Discriminator Network

"""

from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, Dropout


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        d_input = 28*28
        d_output = 1
        
        self.input = Sequential( 
            Linear(d_input, 1024),
            ReLU(),
            Dropout(0.2)
        )
        self.hidden1 = Sequential(
            Linear(1024, 512),
            ReLU(),
            Dropout(0.2)
        )
        self.hidden2 = Sequential(
            Linear(512, 256),
            ReLU(),
            Dropout(0.2)
        )
        self.output = Sequential(
            Linear(256, d_output),
            Sigmoid()
        )
    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x