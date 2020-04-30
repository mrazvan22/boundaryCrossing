import torch.nn as nn
from config import *


class PGGAN():
  
  def __init__():
    pass

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, leaky=True, transpose=False):
  # like the standard convolution, but also adde ReLU layer and defaults to kernel size 3
  if transpose:
    convFunc = nn.ConvTranspose2d
  else:
    convFunc = nn.Conv2d
  
  layers = [convFunc(in_channels, out_channels, kernel_size, stride, padding)]
  
  if leaky:                 
    layers += [nn.LeakyReLU(0.2, inplace=True)]
  else:
    layers += [nn.ReLU()]
  return layers


class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.outResX = 4
    self.outResY = 4
    layers = self.firstBlock() # list of layers
    self.net = nn.Sequential(*layers)
    self.net.add_module('toImage', self.toImage())
    

  def firstBlock(self):
    #input latent vector ngc x 1 x 1
    
    # conv 4x4
    layers = conv(ngc, ngc, kernel_size=4, transpose=True)                
    
    # Conv 3x3, padding of 1
    layers += conv(ngc, ngc, padding=1)            
    return layers

  def toImage(self):
    # to RGB
    return conv(ngc, nc, kernel_size=1)
    

  def grow_network(self):
    print('growing Generator')
    newNet = nn.Sequential()
    print('self.net.named_children',list(self.net.named_children()))
    for name, module in self.net.named_children():
      if name != 'toImage':
        newNet.add_module(name, module) # make new structure
        newNet[-1].load_state_dict(module.state_dict())
    
    # upsample
    newRes = (2*self.outResX, 2*self.ourResY)
    newNet.add_module('upsample', nn.Upsample(size=newRes)

    # Conv 3x3

    # Conv 3x3
    newLayers = conv(ngc, ngc, kernel_size=4, transpose=True)                
    newNet.add_module() 

    self.net = newNet

  def forward(self, input):
    assert input.shape[1] == ngc
    return self.net(input)

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class Discriminator(nn.Module):
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.inRes = 4
    layers = self.fromImage() + self.lastBlock() # list of layers
    self.net = nn.Sequential(*layers)

  def fromImage(self): # like fromRGB but for 1 channel
    # conv 1x1
    return conv(nc, ndc, kernel_size=1)

  def lastBlock(self):
    # conv 3x3, padding=1
    layers = conv(ndc, ndc, padding=1)
      
    # conv 4x4
    layers += conv(ndc, ndc, kernel_size=4)
      
    # fully-connected layer
    layers += [
      nn.Flatten(),
      nn.Linear(in_features=ndc, out_features=1)
    ]
    
    return layers
      


  def forward(self, input):
    assert input.shape[1] == nc
    assert input.shape[2] == input.shape[3]
    return self.net(input)
