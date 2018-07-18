"""
Main Script for building and training the GAN

"""


from torch import optim, save
from torch.nn import BCELoss
from torch.autograd import Variable
import matplotlib.pyplot as plt
from train_func import *
from generator import *
from discriminator import *
from process_data import *


# Load data and create batches
data = loadData('fashion_mnist.csv')
batches = createBatches(data)


# Hyperparameters
lr = 2e-4
epochs = 200
errors_G = []
errors_D = []


# Set up the networks
generator = Generator()
discriminator = Discriminator()
g_optim = optim.Adam(generator.parameters(), lr=lr)
d_optim = optim.Adam(discriminator.parameters(), lr=lr)
loss = BCELoss()


# Train models
print('-------- Training models --------')
for epoch in range(1, epochs):
    for n_batch, batch in enumerate(batches):
        # Train Discriminator
        trueData = Variable(batch)
        falseData = generator(generateNoise(batch.size(0))).detach() # detach to not calculate gradients
        error_d = train_d(d_optim, trueData, falseData)

        # Train Generator
        falseData = generator(generateNoise(batch.size(0)))
        error_g = train_g(g_optim, falseData)
    
    print('Epoch', epoch)
    print('Generator Error:', error_g.data[0])
    print('Discriminator Error:', error_d.data[0])
    print()
    
    errors_G.append(error_g.data[0])
    errors_D.append(error_d.data[0])
print('-------- Finished training --------')


# Save trained models
save(generator.state_dict(), 'generator.pt')
save(discriminator.state_dict(), 'discriminator.pt')


# Plot errors and save figs
figG, axG = plt.subplots()
axG.plot(range(1, epochs), errors_G)
axG.set_xlabel('epoch')
axG.set_ylabel('error')
axG.set_title('Generator')
figG.savefig('errorG.png')


figD, axD = plt.subplots()
axD.plot(range(1, epochs), errors_D)
axD.set_xlabel('epoch')
axD.set_ylabel('error')
axD.set_title('Discriminator')
figD.savefig('errorD.png')