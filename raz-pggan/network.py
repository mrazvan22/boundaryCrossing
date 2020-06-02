import torch.nn as nn
import config
import math
import numpy as np
from torchsummary import summary

class MyDataParallel(nn.DataParallel): # allows parameter fall-though
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class PrintLayer(nn.Module):
  def __init__(self, msg=''):
    super(PrintLayer, self).__init__()
    self.msg = msg

  def forward(self, x):
    print(self.msg,'  shape:', x.shape)
    return x

class PixelNorm(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, eps=1e-08):
    # x is in format BCWH, take mean over channel dim
    return x / ((x ** 2).mean(dim=1, keepdim=True) + eps).sqrt()

class LinearEq(nn.Module):
  def __init__(self, in_features, out_features, leakyParam, equalizeLr):
    super().__init__()    
    self.module = nn.Linear(in_features, out_features)
    size = self.module.weight.size()
    if equalizeLr:
      self.std = math.sqrt(2 / (np.prod(size[1:]) * (1 + leakyParam ** 2)))
    else:
      self.std = 1
  
    nn.init.normal_(self.module.weight, 0.0, 1)
    nn.init.constant_(self.module.bias, 0)


  def forward(self, x):
    return self.std * self.module(x)

class Conv2dEq(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakyParam, equalizeLr):
    super().__init__()
    self.module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    size = self.module.weight.size()
    #print('weight size', size)
    if equalizeLr:
      self.std = math.sqrt(2 / (np.prod(size[1:]) * (1 + leakyParam ** 2)))
    else:
      self.std = 1
      

    nn.init.normal_(self.module.weight, 0.0, 1)
    nn.init.constant_(self.module.bias, 0)
  
  def forward(self, x):
    return self.std * self.module(x)
     
class ConvTranspose2dEq(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakyParam, equalizeLr):
    super().__init__()
    self.module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    size = self.module.weight.size()
    print('weight size', size)
    
    if equalizeLr:
      self.std = math.sqrt(2 / (np.prod(size[1:]) * (1 + leakyParam ** 2)))
    else:
      self.std = 1
    
    nn.init.normal_(self.module.weight, 0.0, 1)
    nn.init.constant_(self.module.bias, 0)
  
  def forward(self, x):
    return self.std * self.module(x)
    

  # like the standard convolution, but also adde ReLU layer and defaults to kernel size 3

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, leaky=True, transpose=False, seq=False, batchNorm=False, layerNorm=False, layerNormRes=None, pixelNorm=False):
  # like the standard convolution, but also adde ReLU layer and defaults to kernel size 3
  if transpose:
    #convFunc = nn.ConvTranspose2d
    convFunc = ConvTranspose2dEq
  else:
    #convFunc = nn.Conv2d
    convFunc = Conv2dEq
  
  
  convObj = convFunc(in_channels, out_channels, kernel_size, stride, padding, config.leakyParam, equalizeLr=config.equalizeLr)
  #nn.init.normal_(convObj.weight, 0.0, 1)
  #nn.init.constant_(convObj.bias, 0)
  layers = [convObj]
  
  if batchNorm:
    batchObj = nn.BatchNorm2d(num_features=out_channels)
    nn.init.normal_(batchObj.weight, 1.0, 0.02)
    nn.init.constant_(batchObj.bias, 0)
    layers += [batchObj]    
  
  if layerNorm:
    layerNormObj = nn.LayerNorm(normalized_shape=(out_channels,layerNormRes,layerNormRes))
    nn.init.normal_(layerNormObj.weight, 1.0, 0.02)
    nn.init.constant_(layerNormObj.bias, 0)
    layers += [layerNormObj]    
  
  if pixelNorm:
    pixelObj = PixelNorm()
    layers += [pixelObj]    

  if leaky:                 
    layers += [nn.LeakyReLU(config.leakyParam, inplace=True)]
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


class GeneratorBlock(nn.Module):
  def __init__(self, gl, nc1, nc2, toImageLayerOld):
    super().__init__()

    self.nc1 = nc1 # input channels
    self.nc2 = nc2 # output channels
    self.gl = gl
    newRes = (config.posResX[self.gl], config.posResY[self.gl])
    self.resX = newRes[0]
    self.resY = newRes[1]

    # input image is already upsampled to the higher resolution
    
    # construct block with two extra layers
    self.block = nn.Sequential()

    # upsample4
    self.block.add_module('upsample%d' % self.gl, nn.Upsample(size=newRes))

    # Conv 3x3
    #self.block.add_module('debug', PrintLayer(msg='GenBlock0'))    
    self.block.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))    

    # Conv 3x3
    #self.block.add_module('debug', PrintLayer(msg='GenBlock1'))    
    self.block.add_module('conv%d_2' % self.gl, conv(self.nc2, self.nc2, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))    
  
    # need to create toImageLayer outside of the block, as at next growth lavel it will be discarded
    #self.block.add_module('debug', PrintLayer(msg='GenBlock2'))    
    self.toImageLayer = self.toImage(inChannels=self.nc2)    

    self.alpha = 0 # blending parameter, starts at 0 and ends at 1

    # for blending, construct parallel block which converts straight to image
    self.noChangeBlock = nn.Sequential()
    # first convert to Image, as the params have already been learned
    self.noChangeBlock.add_module('toImageOld', toImageLayerOld)
    
    # then upsample after toImage. Don't upsample before as that might require re-lertning the toImage convolution
    self.noChangeBlock.add_module('upsample%d' % self.gl, nn.Upsample(size=newRes))



  def forward(self, x):
    # input image is already upsampled to the higher resolution
    xBlock = self.toImageLayer(self.block(x)) # upsampled + passed through new layers
    xImg = self.noChangeBlock(x) # upsampled, from old network
    if config.debug:
      if xBlock.shape[1] != xImg.shape[1] or xBlock.shape[2] != xImg.shape[2] or xBlock.shape[3] != xImg.shape[3]:
        print('xBlock', xBlock.shape)
        print('xImg', xImg.shape)
        raise ValueError('tensor shapes dont match')
    
    return xBlock * self.alpha + xImg * (1 - self.alpha)
  
  def toImage(self, inChannels):
    # to RGB
    layers = conv(inChannels, config.nc, kernel_size=1)
    layers += [nn.Tanh()]
    return nn.Sequential(*layers)
  

class DiscriminatorBlock(nn.Module):
  def __init__(self, gl, nc1, nc2, resX, resY, fromImageLayerOld):
    super().__init__()

    self.nc1 = nc1 # input channels
    self.nc2 = nc2 # output channels
    self.gl = gl
    self.resX = resX
    self.resY = resY

    # convert output to Image
    # need to create fromImageLayer outside of the block, as at next growth lavel it will be discarded
    self.fromImageLayer = self.fromImage(outChannels=self.nc1)


    # input image is already upsampled to the higher resolution
    self.block = nn.Sequential()
    
    # Conv 3x3
    #self.block.add_module('debug', PrintLayer(msg='DiscBlock0'))    
    self.block.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc1, padding=1, seq=True, batchNorm=config.batchNormD, layerNorm=config.layerNormD, layerNormRes=self.resX, pixelNorm=config.pixelNormD))    

    # Conv 3x3
    #self.block.add_module('debug', PrintLayer(msg='DiscBlock1'))    
    self.block.add_module('conv%d_2' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, batchNorm=config.batchNormD, layerNorm=config.layerNormD, layerNormRes=self.resX, pixelNorm=config.pixelNormD))    

    # downsample
    #self.block.add_module('debug', PrintLayer(msg='DiscBlock2'))    
    self.block.add_module('downsample%d' % self.gl, nn.AvgPool2d(kernel_size=2))

    self.noChangeBlock = nn.Sequential()
    self.noChangeBlock.add_module('downsample%d' % self.gl, nn.AvgPool2d(kernel_size=2))
    #self.noChangeBlock.add_module('fromImage', self.fromImage(outChannels=self.nc2))

    # allocate old fromImage layer since that already converged to good solution
    self.noChangeBlock.add_module('fromImage', fromImageLayerOld)
        

    self.alpha = 0 # blending parameter, starts at 0 and ends at 1

  def forward(self, x):
    # input image is already upsampled to the higher resolution
    xBlock = self.block(self.fromImageLayer(x)) # passed through new layers + downsampled
    xImg = self.noChangeBlock(x) # from old network + downsampled
    
    return xBlock * self.alpha + xImg * (1 - self.alpha)
  
  def fromImage(self, outChannels): # like fromRGB but for 1 channel
    # conv 1x1
    return conv(config.nc, outChannels, kernel_size=1, seq=True)

 

 
class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = config.ngpu
    self.gl = 0 # current growth level, add +1 whenever you grow in resolution
    self.nc1 = config.ngc[self.gl] # nr channels before first conv in block (upsample)
    self.nc2 = config.ngc[self.gl] # nr channels after first conv in block (conv1 + conv2)

    self.resX = config.posResX[self.gl]
    self.resY = config.posResX[self.gl]
    
    self.net = nn.Sequential()
    self.net.add_module('firstBlock', self.firstBlock()) 
    self.net.add_module('toImage', self.toImage())

    # wrap in another sequential because we will later pop it during growth. 
    self.net = nn.Sequential(self.net) 

    # reference to latest added block, for updating it's blending parameter
    self.newBlock = None 

  def firstBlock(self):
    #input latent vector ngc x 1 x 1
    block = nn.Sequential()    

    # conv 4x4
    block.add_module('conv1-4x4', conv(self.nc1, self.nc2, kernel_size=4, transpose=True, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))
    
    # Conv 3x3, padding of 1
    block.add_module('conv2-3x3', conv(self.nc2, self.nc2, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))
    return block

  def toImage(self):
    # to RGB
    layers = conv(self.nc2, config.nc, kernel_size=1)
    layers += [nn.Tanh()]
    return nn.Sequential(*layers)
    
  def grow_X_levels(self, extraLevels):
    for l in range(extraLevels):
      self.grow_network()
      self.stopBlending()

  def grow_network(self):
    print('growing Generator')
    #newNet = nn.Sequential()
    self.gl += 1 # add +1 to the growth level
    self.resX = config.posResX[self.gl]
    self.resY = config.posResX[self.gl]
    self.nc1 = config.ngc[self.gl-1] # nr channels before first conv in block (upsample)
    self.nc2 = config.ngc[self.gl] # nr channels after first conv in block (conv1 + conv2)

    # extract the toImage layer from the network to pass it to the new GeneratorBlock below
    toImageLayerOld = self.net[-1][1]
    self.net[-1] = self.net[-1][0]

    #print('self.net.named_children',list(self.net.named_children()))
    #for name, module in self.net.named_children():
    #  newNet.add_module(name, module) # make new structure
    #  newNet[-1].load_state_dict(module.state_dict())
    


    self.newBlock = GeneratorBlock(self.gl, self.nc1, self.nc2, toImageLayerOld)
    self.net.add_module('new_block%d' % self.gl,self.newBlock)

    # convert output to Image
    #newNet.add_module('toImage', self.toImage())

    #self.net = newNet


  def updateAlpha(self, alpha):
    if self.newBlock is not None:
      self.newBlock.alpha = alpha
      print('updated blending parameter:', alpha)
  
  def stopBlending(self):
    if self.newBlock is not None:
      blockWithToImage = nn.Sequential()
      # only take the newBlock.block and toImage layers. Exclude noChangeBlock
      blockWithToImage.add_module('block%d' % self.gl, self.newBlock.block)
      blockWithToImage.add_module('toImage', self.newBlock.toImageLayer)
      self.net[-1] = blockWithToImage
      print('stopped blending')
      print('netG', self.net)
      self.newBlock = None

  def forward(self, input):
    if config.debug:
      assert input.shape[1] == config.latDim
      
      x = input
      print(x.size()) 
      for layer in self.net:
        x = layer(x)
        print(x.size())

      output = x
      if output.shape[2] != self.resX or output.shape[3] != self.resY:
        print('G output shape', output.shape)
        print('G res=(%d, %d)' % (self.resX, self.resY))
        raise ValueError('output dimension in generator does not match')

      return output
    else:
      return self.net(input)


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
    
    # wrap in another sequential because we will later pop it during growth. 
    self.net = nn.Sequential(self.net) 

    # reference to latest added block, for updating it's blending parameter
    self.newBlock = None
    

  def fromImage(self): # like fromRGB but for 1 channel
    # conv 1x1
    return conv(config.nc, self.nc1, kernel_size=1, seq=True)

  def lastBlock(self):
    # conv 3x3, padding=1
    block = nn.Sequential()
    #block.add_module('debug', PrintLayer(msg='DiscLastBlock0'))
    block.add_module('conv1-3x3',conv(self.nc1, self.nc1, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))
      
    # conv 4x4
    #block.add_module('debug', PrintLayer(msg='DiscLastBlock1'))
    block.add_module('conv2-4x4',conv(self.nc2, self.nc2, kernel_size=4, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=1, pixelNorm=config.pixelNormG))
      
    # fully-connected layer

    #block.add_module('debug', PrintLayer(msg='DiscLastBlock1'))
    block.add_module('flatten', nn.Flatten())
    block.add_module('linear', LinearEq(in_features=self.nc2, out_features=1, leakyParam=0, equalizeLr=config.equalizeLr)
)
    block.add_module('sigmoid', nn.Sigmoid())

    #layers += [
    #  nn.Flatten(),
    #  linearObj,
    #  nn.Sigmoid()
    #     ]
    
    return block
      
  def grow_X_levels(self, extraLevels):
    for l in range(extraLevels):
      self.grow_network()
      self.stopBlending()

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

    # pop the fromImageLayer from previous growth level
    #print(list(self.net[0].named_children()))
    assert len(self.net[0]) == 2
    fromImageLayerOld = self.net[0][0]
    self.net[0] = self.net[0][1] # remove from

    self.newBlock = DiscriminatorBlock(self.gl, self.nc1, self.nc2, self.resX, self.resY, fromImageLayerOld)
    newNet.add_module('block%d' % self.gl, self.newBlock)


    print('self.net.named_children',list(self.net.named_children()))
    for name, module in self.net.named_children():
      newNet.add_module(name, module) # make new structure
      newNet[-1].load_state_dict(module.state_dict())

    self.net = newNet


  def updateAlpha(self, alpha):
    if self.newBlock is not None:
      self.newBlock.alpha = alpha

  def stopBlending(self):
    if self.newBlock is not None:
      blockWithFromImage = nn.Sequential()
      # only take the fromImage and newBlock.block layers. Exclude noChangeBlock
      blockWithFromImage.add_module('fromImage', self.newBlock.fromImageLayer)
      blockWithFromImage.add_module('block%d' % self.gl, self.newBlock.block)
      self.net[0] = blockWithFromImage
      print('stopped blending')
      print('netD', self.net)
      self.newBlock = None

  #def removeFromImageLayer(self):
  #  # remove first layer, fromImage, in order to grow the network
  #  self.net[0] = self.net[0][1]

  def forward(self, input):
    
    if config.debug:
      assert input.shape[1] == config.nc
      if input.shape[2] != self.resX or input.shape[3] != self.resY:
        print('D input shape', input.shape)
        print('D res=(%d, %d)' % (self.resX, self.resY))
        raise ValueError('input dimension in discriminator does not match')
      assert input.shape[2] == self.resX
      assert input.shape[3] == self.resY

      x = input
      #print('in tensor size', x.size())
      print('\n\n\n\n--------------\n\n', 'named_modules', [x[0] for x in list(self.net.named_children())] )
      #for name, module in self.net.named_children():
      #print(summary(self.net, (1,self.resX,self.resY)))
      print('nr modules', len(self.net))
      for module in self.net:
        #for layer in module:
        #print('module', name, '  in tensor size:', x.size()) 
        print('in tensor size:', x.size(), '    module', module) 
        assert x.shape[0]
        x = module(x)
        #print('module', name, '  out tensor size:', x.size()) 
      return x
    else:
      return self.net(input)


###########################################################





















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


