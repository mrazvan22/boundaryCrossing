import os
import torch
import matplotlib.pyplot as pl
import pylab
import numpy as np

from pytorch_pretrained_biggan.pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

# load the MNIST models
from gan_pretrained_pytorch.mnist_dcgan.dcgan import Discriminator
from gan_pretrained_pytorch.mnist_dcgan.dcgan import Generator as GeneratorGAN

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import nltk
nltk.download('wordnet')

from abc import ABC, abstractmethod

class ClassifierBC(ABC):

    def __init__(self):
        self.lenZ = 0

    # function below returns scalar from 0 to 1 depending on class probability
    @abstractmethod
    def predict(self, x, targetClass):
        pass



class GeneratorBC(ABC):

    def __init__(self):
        self.lenZ = 0

    @abstractmethod
    def generate(self, z):
        pass


class MNISTGan(GeneratorBC):

    def __init__(self):
        super(MNISTGan, self).__init__()

        self.D = Discriminator(ngpu=1).eval()
        self.G = GeneratorGAN(ngpu=1).eval()

        # load weights
        self.D.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netD_epoch_99.pth'))
        self.G.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netG_epoch_99.pth'))
        if torch.cuda.is_available():
            self.D = self.D.cuda()
            self.G = self.G.cuda()

    def generate(self, z):
        print('z.shape',z.shape)
        convImg = self.convOneImg(z)
        # print(convImg)
        print('convImg.shape', convImg.shape)
        fake_images = self.G(convImg)
        fake_images_np = fake_images.cpu().detach().numpy()
        fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
        return fake_images_np

    def convOneImg(self, z):
        ''' converts the input to the right format for the generator '''
        convImg = z.reshape((z.shape[0], z.shape[1], 1, 1))
        return convImg


# # Load pre-trained model tokenizer (vocabulary)
# model = BigGAN.from_pretrained('biggan-deep-256')
#
# # Prepare a input
# truncation = 0.4
# class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
# noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)
#
# # All in tensors
# noise_vector = torch.from_numpy(noise_vector)
# class_vector = torch.from_numpy(class_vector)
#
# # If you have a GPU, put everything on cuda
# noise_vector = noise_vector.to('cuda')
# class_vector = class_vector.to('cuda')
# model.to('cuda')
#
# # Generate an image
# with torch.no_grad():
#     output = model(noise_vector, class_vector, truncation)
#
# # If you have a GPU put back on CPU
# output = output.to('cpu')
#
# # If you have a sixtel compatible terminal you can display the images in the terminal
# # (see https://github.com/saitoha/libsixel for details)
# # display_in_terminal(output)
#
# # Save results as png images
# save_as_images(output)




# debug = False
# if not debug:
#     num_gpu = 1 if torch.cuda.is_available() else 0
#
#
#
#     D = Discriminator(ngpu=1).eval()
#     G = GeneratorGAN(ngpu=1).eval()
#
#     # load weights
#     D.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netD_epoch_99.pth'))
#     G.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netG_epoch_99.pth'))
#     if torch.cuda.is_available():
#         D = D.cuda()
#         G = G.cuda()

######## generate some images #########

mnistGen = MNISTGan()

batch_size = 1
latent_size = 100

fixed_noise = torch.randn(1, latent_size)
# print('fixed_noise', fixed_noise.shape)
nrImg = 10


latent_prog = torch.tensor(np.zeros((nrImg, latent_size), np.double), dtype=torch.float)
# import pdb
# pdb.set_trace()
latent_prog[0,:] = fixed_noise
for i in range(1,nrImg):
    latent_prog[i, :] = fixed_noise
    latent_prog[i, 0] = 10 * np.double(i)/nrImg


print('latent_prog[0,:,:,:]', latent_prog[0,:])
print('latent_prog[1,:,:,:]', latent_prog[1,:])
print('size latent_prog', latent_prog.shape)

if torch.cuda.is_available():
    latent_prog = latent_prog.cuda()

fake_images_np = mnistGen.generate(latent_prog)


# z = torch.randn(1, latent_size).cuda()
# z = Variable(z)
# fake_images = G(z)

# fake_images_np = fake_images.cpu().detach().numpy()
# fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
R, C = 5, 5
for i in range(nrImg):
    pl.subplot(R, C, i + 1)
    pl.imshow(fake_images_np[i], cmap='gray')

fig = pl.gcf()
pl.show()


######## run discriminator #########

# outputs = D(fake_images)
# print(outputs)
#
# pl.pause(1000)