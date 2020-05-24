import torch.nn as nn
import config


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, leaky=True, transpose=False, seq=False, batchNorm=False, layerNorm=False, layerNormRes=None):
  # like the standard convolution, but also adde ReLU layer and defaults to kernel size 3
  if transpose:
    convFunc = nn.ConvTranspose2d
  else:
    convFunc = nn.Conv2d
  
  convObj = convFunc(in_channels, out_channels, kernel_size, stride, padding)
  nn.init.normal_(convObj.weight, 0.0, 1)
  nn.init.constant_(convObj.bias, 0)
  layers = [convObj]
  
  if batchNorm:
    batchObj = nn.BatchNorm2d(num_features=out_channels)
    nn.init.normal_(batchObj.weight, 1.0, 0.02)
    nn.init.constant_(batchObj.bias, 0)
    layers += [batchObj]    
  
  if layerNorm:
    layerNormObj = nn.LayerNorm(normalized_shape=(out_channels,layerNormRes,layerNormRes))
    layers += [layerNormObj]    

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

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class GeneratorDCGAN3(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorDCGAN3, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config.latDim, config.ngf * 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            # state size. (config.ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            # state size. (config.ngf*4) x 8 x 8
            nn.ConvTranspose2d( config.ngf, config.nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

            #nn.BatchNorm2d(config.ngf * 2),
            #nn.ReLU(True),
            # state size. (config.ngf*2) x 16 x 16
            #nn.ConvTranspose2d( config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(config.ngf),
            #nn.ReLU(True),
            # state size. (config.ngf) x 32 x 32
            #nn.ConvTranspose2d( config.ngf, config.nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# don't apply batchnorm to Disc input layer and Gen output layer, otherwise sample oscillation can occur
class DiscriminatorDCGAN3(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorDCGAN3, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf) x 32 x 32
            #nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(config.ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            
            
            # state size. (config.ndf*2) x 16 x 16
            nn.Conv2d(config.nc, config.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(config.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf) x 4 x 4
            nn.Conv2d(config.ndf, config.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm(normalized_shape=(config.ndf * 2,4,4)),
            #nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*2) x 4 x 4
            nn.Conv2d(config.ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)




class GeneratorDCGAN2(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorDCGAN2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config.latDim, config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.ngf * 8),
            nn.ReLU(True),
            # state size. (config.ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(True),
            # state size. (config.ngf*4) x 8 x 8
            nn.ConvTranspose2d( config.ngf * 4, config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            # state size. (config.ngf*2) x 16 x 16
            nn.ConvTranspose2d( config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            # state size. (config.ngf) x 32 x 32
            nn.ConvTranspose2d( config.ngf, config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorDCGAN2(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorDCGAN2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf) x 32 x 32
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*2) x 16 x 16
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*4) x 8 x 8
            nn.Conv2d(config.ndf * 4, config.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*8) x 4 x 4
            nn.Conv2d(config.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class GeneratorDCGAN(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorDCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config.latDim, config.ngc[0], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(config.ngc[0]),
            nn.ReLU(True),
            # state size. (config.ngf*8) x 4 x 4
            #nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(config.ngf * 4),
            #nn.ReLU(True),
            
            # state size. (config.ngf) x 8 x 8
            nn.ConvTranspose2d( config.ngc[0], config.nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorDCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 4 x 4
            nn.Conv2d(config.nc, config.ndc[-1], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(config.ndc[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*4) x 8 x 8
            #nn.Conv2d(config.ndf * 4, config.ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(config.ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.ndf*8) x 4 x 4
            nn.Conv2d(config.ndc[-1], config.nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


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
    layers = conv(self.nc1, self.nc2, kernel_size=4, transpose=True, batchNorm=config.batchNorm)                
    
    # Conv 3x3, padding of 1
    layers += conv(self.nc2, self.nc2, padding=1, batchNorm=config.batchNorm)
    return layers

  def toImage(self):
    # to RGB
    layers = conv(self.nc2, config.nc, kernel_size=1)
    layers += [nn.Tanh()]
    return nn.Sequential(*layers)
    
  def grow_X_levels(self, extraLevels):
    for l in range(extraLevels):
      self.grow_network()

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
    newNet.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, batchNorm=config.batchNorm))    

    # Conv 3x3
    newNet.add_module('conv%d_2' % self.gl, conv(self.nc2, self.nc2, padding=1, seq=True, batchNorm=config.batchNorm))    

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
    #self.nc1 = config.ndc[config.nrLevels - self.gl - 2] # nr channels before second conv in block
    #self.nc2 = config.ndc[config.nrLevels - self.gl - 1] # nr channels after second conv in block
    self.nc1 = config.ndc[self.gl + 1]
    self.nc2 = config.ndc[self.gl]
    assert self.nc1 <= self.nc2

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
    layers = conv(self.nc1, self.nc1, padding=1, layerNorm=config.layerNorm, layerNormRes=4)
      
    # conv 4x4
    layers += conv(self.nc2, self.nc2, kernel_size=4, layerNorm=config.layerNorm, layerNormRes=1)
      
    # fully-connected layer
    linearObj = nn.Linear(in_features=self.nc2, out_features=1)
    nn.init.normal_(linearObj.weight, 0.0, 1)
    nn.init.constant_(linearObj.bias, 0)

    layers += [
      nn.Flatten(),
      linearObj,
      nn.Sigmoid()
         ]
    
    return layers
      
  def grow_X_levels(self, extraLevels):
    for l in range(extraLevels):
      self.grow_network()

  def grow_network(self):
    print('Growing Discriminator')
    newNet = nn.Sequential()
    self.gl += 1 # add +1 to the growth level
    oldRes = (self.resX, self.resY)
    self.resX = config.posResX[self.gl]
    self.resY = config.posResY[self.gl]

    #self.nc1 = config.ndc[config.nrLevels - self.gl - 2] # before second conv
    #self.nc2 = config.ndc[config.nrLevels - self.gl - 1] # after second conv
    self.nc1 = config.ndc[self.gl + 1]  # before second conv
    self.nc2 = config.ndc[self.gl] # after second conv
    assert self.nc1 <= self.nc2


    # convert output to Image
    newNet.add_module('fromImage', self.fromImage())
    
    # Conv 3x3
    newNet.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc1, padding=1, seq=True, layerNorm=config.layerNorm, layerNormRes=self.resX))    

    # Conv 3x3
    newNet.add_module('conv%d_2' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, layerNorm=config.layerNorm, layerNormRes=self.resX))    

    # downsample
    #newNet.add_module('downsample%d' % self.gl, Downsample(size=oldRes, mode='bilinear'))
    newNet.add_module('downsample%d' % self.gl, nn.AvgPool2d(kernel_size=2))

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
