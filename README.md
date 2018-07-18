# Generative Adversarial Network

Experimenting with Generative Adversarial Networks (GAN) using PyTorch. The GAN built in this project can generate images of clothes.
Here are some of the images it generated:

![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/tshirt2.png "GAN_result")
![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/pants.png "GAN_result")
![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/boot.png "GAN_result")
![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/sandal.png "GAN_result")
![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/shoe.png "GAN_result")
![alt text](https://github.com/alvayliu/Fashion_GAN/blob/master/outputs/tshirt.png "GAN_result")

Data: Fashion MNIST, 60000 samples of greyscale 28x28 px images.


To train the models from scratch, download all python files in this folder and run with  

`python3 train.py`

  


To generate samples with the trained models, download the python files and the models from the trained_models folder. Run with

`python3 test.py`


