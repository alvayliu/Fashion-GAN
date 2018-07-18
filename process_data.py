"""
This file contains functions for loading and generating data

"""

import numpy as np
from torch import Tensor, randn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


def loadData(path):
    data = np.genfromtxt(path, delimiter=',')
    scaler = MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data)
    return Tensor(data)
        
def generateNoise(N):
    noise = Variable(randn(N, 100))
    return noise

def createBatches(data):
    batches = DataLoader(data, batch_size=100, shuffle=True)
    return batches