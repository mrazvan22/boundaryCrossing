import torch
import torch.nn as nn
import config
import math
import numpy as np
from torchsummary import summary
import customFunctions

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
  def __init__(self, in_features, out_features, leakyParam=0, equalizeLr=True, addDim=False):
    super().__init__()    
    self.module = nn.Linear(in_features, out_features)
    self.in_features = in_features
    self.out_features = out_features
    self.addDim = addDim # add singleton dimension at the end, useful as linear activation
    size = self.module.weight.size()
    if equalizeLr:
      self.std = math.sqrt(2 / (np.prod(size[1:]) * (1 + leakyParam ** 2)))
    else:
      self.std = 1
  
    nn.init.normal_(self.module.weight, 0.0, 1)
    nn.init.constant_(self.module.bias, 0)


  def forward(self, x):
    if self.addDim:
      x = x.view(*x.shape, 1)
    #print('LinearEq in_features:', self.in_features)
    #print('LinearEq out_features:', self.out_features)
    #print('LinearEq in shape:', x.shape)
    #print('LinearEq out shape:', (self.std * self.module(x)).shape)
    out = self.std * self.module(x)
    if self.addDim:
      return out.view(*out.shape[:-1])
    else:
      return out

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

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation='leaky', transpose=False, seq=False, batchNorm=False, layerNorm=False, layerNormRes=None, pixelNorm=False):
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

  if activation == 'leaky':
    layers += [nn.LeakyReLU(config.leakyParam, inplace=True)]
  elif activation == 'relu':
    layers += [nn.ReLU()]
  elif activation == 'linear':
    layers += [LinearEq(1, 1, leakyParam=0, equalizeLr=config.equalizeLr, addDim=True)]
  elif activation == 'tanh':
    layers += [nn.Tanh()]
  else:
    raise ValueError('activation can be either leaky, relu, linear or tanh')
    
  if seq:
    return nn.Sequential(*layers)
  else:
    return layers


#class Downsample(nn.Module):
#    def __init__(self, size, mode):
#        super(Downsample, self).__init__()
#        self.size = size
#        self.mode = mode
#        
#    def forward(self, x):
#        return nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=False)

class BatchStdDev(nn.Module):
    def __init__(self):
        super(BatchStdDev, self).__init__()
        
    def forward(self, x):
      # x = BCHW
      #print('x.shape', x.shape)
      var = torch.var(x, dim=0) # variance over batches  512 x 4 x 4
      #print('var.shape', var.shape)
      varMean = torch.mean(x) # mean over channels and spatial locations - scalar
      #print('varMean.shape', varMean.shape)
      std = torch.sqrt(varMean + 1e-08) #
      std = std.view(1,1,1,1) # 1 x 1 x 1 x 1
      std = std.expand(x.shape[0],1,x.shape[2],x.shape[3]) # nrBatches x 1 x 4 x 4
      #print('std shape', std.shape)
      output = torch.cat([x, std], dim=1)
      #print('output shape', output.shape)
      return output


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
    self.block.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))    

    # Conv 3x3
    self.block.add_module('conv%d_2' % self.gl, conv(self.nc2, self.nc2, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))    
  
    # need to create toImageLayer outside of the block, as at next growth lavel it will be discarded
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
    return conv(inChannels, config.nc, kernel_size=1, activation = config.activationFinal, seq = True)
  

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
    self.block.add_module('conv%d_1' % self.gl, conv(self.nc1, self.nc1, padding=1, seq=True, batchNorm=config.batchNormD, layerNorm=config.layerNormD, layerNormRes=self.resX, pixelNorm=config.pixelNormD))    

    # Conv 3x3
    self.block.add_module('conv%d_2' % self.gl, conv(self.nc1, self.nc2, padding=1, seq=True, batchNorm=config.batchNormD, layerNorm=config.layerNormD, layerNormRes=self.resX, pixelNorm=config.pixelNormD))    

    # downsample
    self.block.add_module('downsample%d' % self.gl, nn.AvgPool2d(kernel_size=2))

    self.noChangeBlock = nn.Sequential()
    self.noChangeBlock.add_module('downsample%d' % self.gl, nn.AvgPool2d(kernel_size=2))

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
  def __init__(self):
    super(Generator, self).__init__()
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

    self.device = "cpu"

    
  def set(self, device):
    self.device = device
    self.to(device)

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
    return conv(self.nc2, config.nc, kernel_size=1, activation = config.activationFinal, seq = True)
    
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
    self.net[-1] = self.net[-1][0] # remove toImage layer

    #print('self.net.named_children',list(self.net.named_children()))
    #for name, module in self.net.named_children():
    #  newNet.add_module(name, module) # make new structure
    #  newNet[-1].load_state_dict(module.state_dict())
    


    self.newBlock = GeneratorBlock(self.gl, self.nc1, self.nc2, toImageLayerOld)
    self.net.add_module('new_block%d' % self.gl,self.newBlock)

    # convert output to Image
    #newNet.add_module('toImage', self.toImage())

    #self.net = newNet
    self.to(self.device)


  def updateAlpha(self, alpha):
    if self.newBlock is not None:
      self.newBlock.alpha = alpha
      print('updated blending parameter: %.2f' % alpha)
  
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

  def forward(self, x):
    if config.debug:
      assert x.shape[1] == config.latDim
      
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
      return self.net(x)

   

class ModelParallel(nn.Module):
  def __init__(self, net, split_size, gpus):
    super(ModelParallel, self).__init__()
    self.net = net # currently on cpu
    self.split_size = split_size # size of micro-batch 
    self.gpus = gpus
    self.create() # create gpu modules and rest of object

  def create(self): # needs to be called whenever we grow/modify self.net  
    self.subnet = self.net.net
    self.modules = list(self.subnet.named_children())
    self.names = [x[0] for x in self.modules]
    self.modules = [x[1] for x in self.modules]
    self.ngpus = len(self.gpus)
    
    # allocate modules to gpu according to the number of parameters in each module
    paramsPerModule = [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in self.modules]
    print('paramsPerModule', paramsPerModule)
    print('ngpus', self.ngpus)
    paramsPartitioned = customFunctions.partition_list(paramsPerModule, self.ngpus)
    print('paramsPartitioned', paramsPartitioned)
    nrModulesPerGpu = [len(x) for x in paramsPartitioned]
    c = 0
    self.gpuModules = [[] for g in range(self.ngpus)]
    for g in range(self.ngpus):
      for m in range(nrModulesPerGpu[g]):
        self.gpuModules[g] += [self.modules[c]]
        c += 1

    print('gpuModules', self.gpuModules)

    # create containers from lists and move to different gpus
    for m in range(len(self.gpuModules)):
      self.modules[m] = nn.Sequential(*self.gpuModules[m]).to(self.gpus[m])
  

  def forward(self, x):

    if self.ngpus == 1:
      return self.modules[0](x)
    elif self.ngpus == 2:

      splits = iter(x.split(self.split_size, dim=0))
      s_next = next(splits) # dimension split_size x C x W x H
      

      s_prev = self.modules[0](s_next).to(self.gpus[1])
      ret = []

      for s_next in splits:
          # A. s_prev runs on cuda:1
          s_prev = self.modules[1](s_prev)
          ret.append(s_prev)

          # B. s_next runs on cuda:0, which can run concurrently with A
          s_prev = self.modules[0](s_next).to(self.gpus[1])

      s_prev = self.modules[1](s_prev)
      ret.append(s_prev)

      return torch.cat(ret)

  #def __getattr__(self, name):
  #    if hasattr(self.net, name):
  #      #  return self.net.name
  #      return getattr(self.net, name)
      #except AttributeError:
      #else:  
      #  return self.__dict__.get(name)

  def updateAlpha(self, *args, **kwargs): return self.net.updateAlpha(*args, **kwargs)
  def stopBlending(self, *args, **kwargs): return self.net.stopBlending(*args, **kwargs)
  
  def grow_network(self, *args, **kwargs): 
    self.net.grow_network(*args, **kwargs)
    self.create() # re-create the object, as gpu allocations might be different now
    

        
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
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
    
    self.device = "cpu" # by default it is built on cpu
    
  def set(self, device):
    self.device = device
    self.to(device)

  def fromImage(self): # like fromRGB but for 1 channel
    # conv 1x1
    return conv(config.nc, self.nc1, kernel_size=1, seq=True)

  def lastBlock(self):
    # conv 3x3, padding=1
    block = nn.Sequential()

    block.add_module('batchStdDev',BatchStdDev())

    block.add_module('conv1-3x3',conv(self.nc1+1, self.nc1, padding=1, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=self.resX, pixelNorm=config.pixelNormG))
      
    # conv 4x4
    block.add_module('conv2-4x4',conv(self.nc2, self.nc2, kernel_size=4, stride=1, padding=0, seq=True, batchNorm=config.batchNormG, layerNorm=config.layerNormG, layerNormRes=1, pixelNorm=config.pixelNormG))
      
    # fully-connected layer
    block.add_module('flatten', nn.Flatten())
    block.add_module('fully-connected', LinearEq(in_features=self.nc2, out_features=config.nc, leakyParam=0, equalizeLr=config.equalizeLr))

    # add activation  
    if config.activationFinal == 'linear':  
      block.add_module('linear-activation', LinearEq(in_features=config.nc, out_features=config.nc, leakyParam=0, equalizeLr=config.equalizeLr)
)
    elif config.activationFinal == 'tanh':
      block.add_module('tanh', nn.Tanh())
    elif config.activationFinal == 'leaky':
      block.add_module('tanh', nn.LeakyReLU(0.2))
    elif config.activationFinal == 'relu':
      block.add_module('tanh', nn.ReLU(True))
    else:
      raise ValueError('activation can be either leaky, relu, linear or tanh')
    
    #block.add_module('sigmoid', nn.Sigmoid())

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


    #print('self.net.named_children',list(self.net.named_children()))
    for name, module in self.net.named_children():
      newNet.add_module(name, module) # make new structure
      newNet[-1].load_state_dict(module.state_dict())

    self.net = newNet

    self.to(self.device)


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

