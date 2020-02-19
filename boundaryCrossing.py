import os
import torch
import matplotlib.pyplot as pl
import pylab
import numpy as np

import torch.nn as nn
import torch.optim as optim

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
    self.lenZ = 0 # size/dimensionality of latent space (as integer)

  @abstractmethod
  def generate(self, z):
    pass

  @abstractmethod
  def invertImage(self, x):
    pass


class MNISTGan(GeneratorBC):
  ''' MNIST GAN visualiser that implements the Generator interface. Add a new class for every
  new dataset that needs to be visualised. '''
  def __init__(self):
    super(MNISTGan, self).__init__()

    # self.D = Discriminator(ngpu=1).eval()
    self.G = GeneratorGAN(ngpu=1).eval()

    self.lenZ = 100

    # load weights
    # self.D.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netD_epoch_99.pth'))
    self.G.load_state_dict(torch.load('gan_pretrained_pytorch/mnist_dcgan/weights/netG_epoch_99.pth'))
    if torch.cuda.is_available():
        # self.D = self.D.cuda()
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

  def invertImage(self, x):
    inputRand = torch.randn(1, self.lenZ, 1, 1)
    inputRand = inputRand.cuda()
    output = self.G(inputRand)
    loss = torch.nn.L1Loss()
    lossValue = loss(x,inputRand)
    grad = torch.autograd.grad(lossValue, inputRand)
    print(inputRand.shape, grad.shape)
    adsa



class VisualiserBC():
  ''' Visualiser class. Given a generator and discriminator, estimates a sequence of optimal latent points
  that cross the boundary. '''
  def __init__(self, generator, discriminator):
      self.generator = generator
      self.discriminator = discriminator



  def estimZpoints(self, initImg):
      initZ = self.generator.invertImage(initImg)

      randLatent = torch.randn(1, self.generator.lenZ)
      # print('fixed_noise', fixed_noise.shape)
      nrImg = 10

      progLatent = torch.tensor(np.zeros((nrImg, self.generator.lenZ), np.double), dtype=torch.float)
      progLatent[0, :] = randLatent
      for i in range(1, nrImg):
          progLatent[i, :] = randLatent
          progLatent[i, 0] = 10 * np.double(i) / nrImg


      loss_f = nn.L1Loss()
      optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

      print('latent_prog[0,:,:,:]', progLatent[0, :])
      print('latent_prog[1,:,:,:]', progLatent[1, :])
      print('size latent_prog', progLatent.shape)

      if torch.cuda.is_available():
          progLatent = progLatent.cuda()

      return progLatent


  def showZpoints(self, progLatent, initImg):
      nrImg = progLatent.shape[0]
      fake_images_np = self.generator.generate(progLatent)

      R, C = 3, 5
      pl.subplot(R, C, 1)
      pl.imshow(initImg, cmap='gray')

      for i in range(nrImg):
          pl.subplot(R, C, i + 2)
          pl.imshow(fake_images_np[i], cmap='gray')


      fig = pl.gcf()
      pl.show()

  def visualise(self, initImg):
      progLatent = self.estimZpoints(initImg)
      # self.showZpoints(progLatent, initImg)





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

# mnistGen = MNISTGan()




######## run discriminator #########

# outputs = D(fake_images)
# print(outputs)
#
# pl.pause(1000)