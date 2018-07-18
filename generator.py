"""
This class defines the Generator Network

"""

from torch.nn import Module, Sequential, Linear, ReLU, Tanh, Dropout


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        d_input = 100
        d_output = 28*28
        
        self.input = Sequential(
            Linear(d_input, 256),
            ReLU()
        )
        self.hidden1 = Sequential(            
            Linear(256, 512),
            ReLU()
        )
        self.hidden2 = Sequential(
            Linear(512, 1024),
            ReLU()
        )
        self.output = Sequential(
            Linear(1024, d_output),
            Tanh()
        )
        
    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x