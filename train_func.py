"""
Functions for training the Generator and Discriminator

"""

from torch import ones, zeros
from torch.autograd import Variable



def train_d(opt, trueData, falseData):
    trueN = trueData.size(0)
    falseN = falseData.size(0)
    opt.zero_grad()
        
    # Train on true data
    truePred = discriminator(trueData)
    trueError = loss(truePred, Variable(ones(trueN, 1)))
    trueError.backward()

    # Train on generated data
    falsePred = discriminator(falseData)
    falseError = loss(falsePred, Variable(zeros(falseN, 1)))
    falseError.backward()
    
    opt.step()
    return trueError + falseError



def train_g(opt, falseData):
    falseN = falseData.size(0)
    opt.zero_grad()

    # Get response from Discriminator
    pred = discriminator(falseData) 

    # Pretend that the false data is true data
    error = loss(pred, Variable(ones(falseN, 1)))
    error.backward()

    opt.step()
    return error