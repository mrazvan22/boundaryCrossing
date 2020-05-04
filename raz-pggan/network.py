import torch.nn as nn
import config


class PGGAN():
  
  def __init__():
    pass

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, leaky=True, transpose=False, seq=False):
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

  if seq:
    return nn.Sequential(*layers)
  else:
    return layers


class Downsample(nn.Module):
    def __init__(self, size, mode):
        super(Downsample, self).__init__()
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=False)


class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = config.ngpu
    self.gl = 0 # current growth level, add +1 whenever you grow in resolution
    self.nc1 = config.ngc[self.gl] # nr channels before first conv in block (upsample)
    self.nc2 = config.ngc[self.gl] # nr channels after first conv in block (conv1 + conv2)

    self.resX = config.posResX[self.gl]
    self.resY = config.posResX[self.gl]
    layers = self.firstBlock() # list of layers
    self.net = nn.Sequential(*layers)
    self.net.add_module('toImage', self.toImage())

  def firstBlock(self):
    #input latent vector ngc x 1 x 1
    
    # conv 4x4
    layers = conv(self.nc1, self.nc2, kernel_size=4, transpose=True)                
    
    # Conv 3x3, padding of 1
    layers += conv(self.nc2, self.nc2, padding=1)
    return layers

  def toImage(self):
    # to RGB
    return conv(self.nc2, config.nc, kernel_size=1, seq=True)
    

  def grow_network(self):
    print('growing Generator')
    newNet = nn.Sequential()
    self.gl += 1 # add +1 to the growth level
    self.resX = config.posResX[self.gl]
    self.resY = config.posResX[self.gl]
    self.nc1 = config.ngc[self.gl-1] # nr channels before first conv in block (upsample)
    self.nc2 = config.ngc[self.gl] # nr channels after first conv in block (conv1 + conv2)

    print('self.net.named_children',list(self.net.named_children()))
    for name, module in self.net.named_children():
      if name != 'toImage':
        newNet.add_module(name, module) # make new structure
        newNet[-1].load_state_dict(module.state_dict())
    
    # upsample
    newRes = (config.posResX[self.gl], config.posResY[self.gl])
    newNet.add_module('upsample%d' % self.gl, nn.Upsample(size=newRes))

    # Conv 3x3
    newNet.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True))    

    # Conv 3x3
    newNet.add_module('conv%d_2' % self.gl, conv(self.nc2, self.nc2, padding=1, seq=True))    

    # convert output to Image
    newNet.add_module('toImage', self.toImage())

    self.net = newNet

  def forward(self, input):
    assert input.shape[1] == config.latDim
    output = self.net(input)
    if output.shape[2] != self.resX or output.shape[3] != self.resY:
      print('G output shape', output.shape)
      print('G res=(%d, %d)' % (self.resX, self.resY))
      raise ValueError('output dimension in generator does not match')
    return output

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class Discriminator(nn.Module):
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = config.ngpu
    self.gl = 0 # current growth level, add +1 whenever you grow in resolution
    self.nc1 = config.ndc[config.nrLevels - self.gl - 2] # nr channels before second conv in block
    self.nc2 = config.ndc[config.nrLevels - self.gl - 1] # nr channels after second conv in block

    self.resX = config.posResX[self.gl]
    self.resY = config.posResX[self.gl]

    self.net = nn.Sequential()
    self.net.add_module('fromImage', self.fromImage())
    self.net.add_module('lastBlock', nn.Sequential(*self.lastBlock()))
    

  def fromImage(self): # like fromRGB but for 1 channel
    # conv 1x1
    return conv(config.nc, self.nc1, kernel_size=1, seq=True)

  def lastBlock(self):
    # conv 3x3, padding=1
    layers = conv(self.nc1, self.nc1, padding=1)
      
    # conv 4x4
    layers += conv(self.nc2, self.nc2, kernel_size=4)
      
    # fully-connected layer
    layers += [
      nn.Flatten(),
      nn.Linear(in_features=self.nc2, out_features=1)
    ]
    
    return layers
      
  def grow_network(self):
    print('Growing Discriminator')
    newNet = nn.Sequential()
    self.gl += 1 # add +1 to the growth level
    oldRes = (self.resX, self.resY)
    self.resX = config.posResX[self.gl]
    self.resY = config.posResY[self.gl]

    self.nc1 = config.ndc[config.nrLevels - self.gl - 2] # nr channels in first layer
    self.nc2 = config.ndc[config.nrLevels - self.gl - 1] # nr channels in first layer

    # convert output to Image
    newNet.add_module('fromImage', self.fromImage())
    
    # Conv 3x3
    newNet.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc1, padding=1, seq=True))    

    # Conv 3x3
    newNet.add_module('conv%d_2' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True))    

    # downsample
    newNet.add_module('downsample%d' % self.gl, Downsample(size=oldRes, mode='bilinear'))

    print('self.net.named_children',list(self.net.named_children()))
    for name, module in self.net.named_children():
      if name != 'fromImage':
        newNet.add_module(name, module) # make new structure
        newNet[-1].load_state_dict(module.state_dict())

    self.net = newNet



  def forward(self, input):
    assert input.shape[1] == config.nc
    if input.shape[2] != self.resX or input.shape[3] != self.resY:
      print('D input shape', input.shape)
      print('D res=(%d, %d)' % (self.resX, self.resY))
      raise ValueError('input dimension in discriminator does not match')
    assert input.shape[2] == self.resX
    assert input.shape[3] == self.resY
    return self.net(input)
