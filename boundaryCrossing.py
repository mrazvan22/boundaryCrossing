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

torch.manual_seed(3)

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

  @abstractmethod
  def parameters(self):
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
    fake_images = self.__call__(z)
    fake_images_np = fake_images.cpu().detach().numpy()
    return fake_images_np.reshape(fake_images_np.shape[0], 28, 28)


  def convOneImg(self, z):
    ''' converts the input to the right format for the generator '''
    convImg = z.view((z.shape[0], z.shape[1], 1, 1))
    return convImg

  def parameters(self):
    return self.G.parameters()

  def invertImage(self, x):
    xCuda = x.cuda()
    zCurr = torch.randn(1, self.lenZ)
    zCurr = zCurr.cuda()

    zCurr.requires_grad_()
    for param in self.G.parameters():
      param.requires_grad = False

    optimizer = optim.Adam([zCurr], lr=0.05)

    for i in range(400):
      optimizer.zero_grad()
      output = self.G(self.convOneImg(zCurr))
      loss = torch.nn.L1Loss()
      lossValue = loss(xCuda, output)
      lossValue.backward()

      optimizer.step()

    # self.showZpoints(zCurr, x)

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
    pl.imshow(initImg.cpu(), cmap='gray')
    pl.gca().title.set_text('Original')

    for i in range(nrImg):
      pl.subplot(R, C, i + 2)
      pl.imshow(fake_images_np[i], cmap='gray')
      pl.gca().title.set_text('n = %d' % (i+1))

    fig = pl.gcf()
    pl.show()



class VisualiserBC(object):
  ''' Visualiser class. Given a generator and discriminator, estimates a sequence of optimal latent points
  that cross the boundary. '''
  def __init__(self, generator, discriminator, targetExemplarImgs):
    self.generator = generator

    # discriminator(image_x) is a scalar, where higher number represents target class.
    self.discriminator = discriminator

    self.targetExemplarImgs = targetExemplarImgs
    self.targetExemplarLatents = torch.empty((targetExemplarImgs.shape[0], self.generator.lenZ),
                                             requires_grad=False)
    for i in range(targetExemplarImgs.shape[0]):
      self.targetExemplarLatents[i,:] = self.generator.invertImage(targetExemplarImgs[i,:,:])

    self.targetExemplarLatents = self.targetExemplarLatents.new_tensor(
      data=self.targetExemplarLatents.data.numpy(), requires_grad=False, device='cuda')

    self.nrTargetExemplars = self.targetExemplarLatents.shape[0]


  def initProgLatent(self, initImg):
    initZ = self.generator.invertImage(initImg)
    nrProgImgs = 3

    # progLatent = torch.tensor(np.zeros((nrImg, self.generator.lenZ), np.double), dtype=torch.float, requires_grad=True)
    progLatent = torch.empty((nrProgImgs, self.generator.lenZ))

    # zCurr.requires_grad_()
    # for param in self.G.parameters():
    #   param.requires_grad = False

    # print('self.generator(initZ).shape', self.generator(initZ).shape)
    # asda

    predScore = self.discriminator(self.generator(initZ))
    print(predScore)

    # loss = torch.nn.L1Loss()
    # lossValue = loss(xCuda, predScore)
    predScore.backward()

    stepSize = 0.2
    # progLatent[0, :] = initZ
    for i in range(nrProgImgs):
      progLatent[i, :] = initZ - initZ.grad.data * stepSize * i

    progLatent = progLatent.new_tensor(data=progLatent.data.numpy(),
                                       requires_grad=True, device='cuda')
    print(progLatent)
    # asda

    # if torch.cuda.is_available():
    #   progLatent = progLatent.cuda()

    return progLatent

  def estimZpoints(self, initImg):

    outFld = 'generated/mnistGan'
    os.system('mkdir -p %s' % outFld)

    initImg = initImg.cuda()
    # targetImg = targetImg.cuda()
    progLatentPL = self.initProgLatent(initImg)
    # initLatent = progLatent[0,:].clone().detach().requires_grad_(False)
    initLatent = self.generator.invertImage(initImg).requires_grad_(False)
    # initLatent = initLatent.new_tensor(data=initLatent.cpu().data.numpy(),
    #                                    requires_grad=False, device='cuda')
    # print(initLatent.grad)
    # asd
    nrProgImgs = progLatentPL.shape[0]
    # targetImgVectPTL - prog_images x target_images x lat_dim
    targetsLatentPTL = self.targetExemplarLatents.repeat(nrProgImgs, 1, 1)
    initImgVectPDD = initImg.repeat(nrProgImgs, 1, 1)
    initLatentVectPL = initLatent.repeat(nrProgImgs, 1)

    progLatentPL.requires_grad = True
    for param in self.generator.parameters():
      param.requires_grad = False

    for param in self.discriminator.parameters():
      param.requires_grad = False

    # progLatent = progLatent.cuda()
    optimizer = optim.SGD([progLatentPL], lr=0.1, momentum=0.9)
    lambdaF = 1
    lambdaId = 0
    lambdaTarget = 0.1
    lossF = nn.L1Loss() # actually MAE loss
    lossId = nn.L1Loss()
    lossTarget = nn.L1Loss()

    predScores = self.discriminator(self.generator(progLatentPL), printShape=False)
    targetScores = torch.tensor(np.array(range(1,nrProgImgs+1)) / nrProgImgs, dtype=predScores.dtype).cuda()
    print('targetScores', targetScores)
    print('self.targetExemplarImgs.shape', self.targetExemplarImgs.shape)

    for t in range(self.nrTargetExemplars):
      print('disc(targets):', self.discriminator(self.targetExemplarImgs[t,:,:].view(
        1,1,self.targetExemplarImgs.shape[1], self.targetExemplarImgs.shape[2]).cuda()))
      print('disc(targetLatent):', self.discriminator(self.generator(self.targetExemplarLatents[t, :])))

    print('disc(initImg):', self.discriminator.model(initImg.view(1,1,initImg.shape[0], initImg.shape[1]).cuda()))
    print('discNorm(initImg):', self.discriminator(initImg.view(1,1,initImg.shape[0], initImg.shape[1]).cuda()))

      # print('loss two targets:', lossTarget(targetsLatentPTL[0,0,:], targetsLatentPTL[0,t,:]))

    # for t in range(self.nrTargetExemplars):
    #   print('loss latent targets:', lossTarget(progLatentPL[0,:], targetsLatentPTL[0,t,:]))

    self.visDirectTransitionInitTarget(initLatent, self.targetExemplarLatents[0,:].view(1,-1))

    lossTargetVals = [0 for x in range(self.nrTargetExemplars)]

    nrIter = 1000
    progZeroTraceIL = torch.empty((nrIter, self.generator.lenZ), requires_grad = False, device = 'cuda')

    for i in range(nrIter):

      progZeroTraceIL[i,:] = progLatentPL[0,:]
      # self.showZpoints(progLatent, initImg)
      genImgs = self.generator(progLatentPL)
      predScores = self.discriminator(genImgs, printShape=False)

      # print('targetScores', targetScores.shape)
      # print('predScores', predScores.shape)
      # print()
      # make sure the predicted scores are [0, ]

      # print(initImgVect.shape)
      # assert initImgVect.shape[0] == nrImg
      # assert initImgVect.shape[1] == initImg.shape[0]
      lossFVal = lossF(predScores, targetScores)
      lossIdVal = lossId(initImgVectPDD, genImgs)
      # print(initImgVectPDD.shape)
      # print(targetImgVectPTL[:,0,:].shape)
      for t in range(self.nrTargetExemplars):
        lossTargetVals[t] = lossTarget(progLatentPL, targetsLatentPTL[:,t,:])

      lossTargetValMean = torch.mean(torch.stack(lossTargetVals))

      print('i=%d ---------' % i)
      print('predScores', predScores)
      print('lossFVal', lossFVal)
      if lambdaId > 0:
        print('lossIdVal', lossIdVal)
      print('lossTargetVals', lossTargetVals)
      print('lossTargetValMean', lossTargetValMean)
      # print('lambda_id * lossIdVal', lambdaId * lossIdVal)

      # lossVal = lossFVal + lambdaId * lossIdVal + lambdaTarget * lossTargetVal
      lossVal = lambdaF * lossFVal + lambdaId * lossIdVal + lambdaTarget * lossTargetValMean

      print('lossVal', lossVal)

      optimizer.zero_grad()
      lossVal.backward()
      optimizer.step()

      if (i % (nrIter/100)) == 0:
        pass
        self.showZpoints(progLatentPL, initImg,
                       fileName = '%s/mnist_%.3d' % (outFld, i))

    return progLatentPL

  def visDirectTransitionInitTarget(self, initLatent, targetLatent):
    nrImg = 10
    initImg = self.generator.genAsImg(initLatent)[0,:,:]
    targetImg = self.generator.genAsImg(targetLatent)[0,:,:]
    initScore = self.discriminator(self.generator(initLatent))
    targetScore = self.discriminator(self.generator(targetLatent))

    R, C = 3, 5
    pl.subplot(R, C, 1)
    pl.imshow(initImg, cmap='gray')
    pl.gca().title.set_text('Init = %.3f' % initScore)
    pl.subplot(R, C, 2)
    pl.imshow(targetImg, cmap='gray')
    pl.gca().title.set_text('Target = %.3f' % targetScore)

    vectorDir = targetLatent - initLatent

    for i in range(nrImg):
      inBetweenLat = initLatent + vectorDir * float(i)/nrImg
      inBetweenImg = self.generator.genAsImg(inBetweenLat)[0,:,:]
      predScore = self.discriminator(self.generator(inBetweenLat))

      pl.subplot(R, C, i + 3)
      pl.imshow(inBetweenImg, cmap='gray')
      pl.gca().title.set_text('s = %f' % predScore)


    fig = pl.gcf()
    pl.show()

  def showZpoints(self, progLatent, initImg, fileName = None):
    nrImg = progLatent.shape[0]
    fake_images_np = self.generator.genAsImg(progLatent)

    R, C = 3, 5
    pl.subplot(R, C, 1)
    pl.imshow(initImg.cpu(), cmap='gray')
    pl.gca().title.set_text('Original')

    for i in range(nrImg):
      pl.subplot(R, C, i + 2)
      pl.imshow(fake_images_np[i], cmap='gray')
      pl.gca().title.set_text('n = %d' % (i+1))

    for i in range(self.nrTargetExemplars):
      pl.subplot(R, C, i + nrImg + 2)
      pl.imshow(self.targetExemplarImgs[i].cpu(), cmap='gray')
      pl.gca().title.set_text('target %d' % (i+1))

    fig = pl.gcf()
    if fileName is not None:
      fig.savefig(fileName, dpi=150)
    else:
      pl.show()

  def visualise(self, initImg):
    progLatent = self.estimZpoints(initImg)
    self.showZpoints(progLatent, initImg)




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