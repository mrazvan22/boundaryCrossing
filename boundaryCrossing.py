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
  def __call__(self, z):
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

  def __call__(self, z):
    return self.G(self.convOneImg(z))

  def genAsImg(self, z):
    fake_images = self.forward(z)
    fake_images_np = fake_images.cpu().detach().numpy()
    return fake_images_np.reshape(fake_images_np.shape[0], 28, 28)


  def convOneImg(self, z):
    ''' converts the input to the right format for the generator '''
    convImg = z.view((z.shape[0], z.shape[1], 1, 1))
    return convImg

  def invertImage(self, x):
    xCuda = x.cuda()
    torch.manual_seed(3)
    zCurr = torch.randn(1, self.lenZ)
    zCurr = zCurr.cuda()

    zCurr.requires_grad_()
    for param in self.G.parameters():
      param.requires_grad = False

    optimizer = optim.Adam([zCurr], lr=0.05)

    for i in range(1000):
      optimizer.zero_grad()
      output = self.G(self.convOneImg(zCurr))
      loss = torch.nn.L1Loss()
      lossValue = loss(xCuda, output)
      lossValue.backward()

      optimizer.step()

    return zCurr

  # def invertImage(self, x):
  #   xCuda = x.cuda()
  #   torch.manual_seed(3)
  #   zCurr = torch.randn(1, self.lenZ)
  #   zCurr = zCurr.cuda()
  #
  #   # self.showZpoints(zCurr, x)
  #
  #   zCurr.requires_grad_()
  #   for param in self.G.parameters():
  #     param.requires_grad = False
  #
  #   # optimizer = optim.SGD(zCurr, lr=5)
  #
  #   import sys
  #   learning_rate = 5
  #   for i in range(1000):
  #     self.G.zero_grad()
  #     output = self.G(self.convOneImg(zCurr))
  #     loss = torch.nn.L1Loss()
  #     lossValue = loss(xCuda,output)
  #     print('lossValue', lossValue)
  #
  #     lossValue.backward()
  #
  #     with torch.no_grad():
  #       # -= performs in-place update. This is important as doing
  #       # z = z - grad * lr doesn't work, since it zeroes the gradient.
  #       zCurr -= zCurr.grad.data * learning_rate
  #
  #     zCurr.grad.data.zero_()
  #
  #   return zCurr

  def showZpoints(self, progLatent, initImg):
    nrImg = progLatent.shape[0]
    fake_images_np = self.genAsImg(progLatent)

    R, C = 3, 5
    pl.subplot(R, C, 1)
    pl.imshow(initImg, cmap='gray')
    pl.gca().title.set_text('Original')

    for i in range(nrImg):
      pl.subplot(R, C, i + 2)
      pl.imshow(fake_images_np[i], cmap='gray')
      pl.gca().title.set_text('n = %d' % (i+1))

    fig = pl.gcf()
    pl.show()



class VisualiserBC():
  ''' Visualiser class. Given a generator and discriminator, estimates a sequence of optimal latent points
  that cross the boundary. '''
  def __init__(self, generator, discriminator):
    self.generator = generator

    # discriminator(image_x) is a scalar, where higher number represents target class.
    self.discriminator = discriminator


  def initProgLatent(self, initImg):
    initZ = self.generator.invertImage(initImg)
    nrImg = 10

    progLatent = torch.tensor(np.zeros((nrImg, self.generator.lenZ), np.double), dtype=torch.float)

    # zCurr.requires_grad_()
    # for param in self.G.parameters():
    #   param.requires_grad = False

    predScore = self.discriminator(self.generator(initZ))
    print(predScore)
    
    # loss = torch.nn.L1Loss()
    # lossValue = loss(xCuda, predScore)
    predScore.backward()

    stepSize = 0.1
    progLatent[0, :] = initZ
    for i in range(1, nrImg):
      progLatent[i, :] = initZ - initZ.grad.data * stepSize

    if torch.cuda.is_available():
      progLatent = progLatent.cuda()

    return progLatent

  def estimZpoints(self, initImg):

    progLatent = self.initProgLatent(initImg)

    self.showZpoints(progLatent, initImg)

    loss_f = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('latent_prog[0,:,:,:]', progLatent[0, :])
    print('latent_prog[1,:,:,:]', progLatent[1, :])
    print('size latent_prog', progLatent.shape)

    return progLatent


  def showZpoints(self, progLatent, initImg):
    nrImg = progLatent.shape[0]
    fake_images_np = self.generator.generate(progLatent)

    R, C = 3, 5
    pl.subplot(R, C, 1)
    pl.imshow(initImg, cmap='gray')
    pl.gca().title.set_text('Original')

    for i in range(nrImg):
      pl.subplot(R, C, i + 2)
      pl.imshow(fake_images_np[i], cmap='gray')
      pl.gca().title.set_text('n = %d' % (i+1))

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