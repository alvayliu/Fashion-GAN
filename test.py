"""
Main Script for loading and testing trained models

"""

from torch import load
from torchvision import transforms
import matplotlib.pyplot as plt
from generator import *
from discriminator import *
from process_data import *

generator = Generator()
generator.load_state_dict(load('trained_models/generator.pt'))

testData = generator(generateNoise(10))

for image in testData:
    image = image.view(28,28).data
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.show()