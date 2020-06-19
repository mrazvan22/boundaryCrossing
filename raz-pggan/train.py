#from __future__ import print_fuconfig.nction
#%matplotlib inline
#import argparse
import os
import sys
import random
import torch
#from torch import randn, full, no_grad

import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from IPython.display import HTML
import time
#from cvtorchvision import cvtransforms
import importlib
import pickle
import socket

#from config import *
import config
import DataLoaderOptimised
import network



# Set random seed for reproducibility
manualSeed = 4
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
sys.stdout.flush()

def allocGpus(gpus):
  '''returns 2 lists (gpusG, gpusD) denoting the available gpus for Generator and Discriminator. Generally allocates more gpus to discriminator '''
  allocs = [
    (), 
    ([0],[0]), 
    ([0],[1]), 
    ([0],[1,2]), 
    ([0,1],[2,3]), 
    ([0,1],[2,3,4]),
    ([0,1,2],[3,4,5]),
    ([0,1,2],[3,4,5,6]),
    ([0,1,2,3],[4,5,6,7]),
    ([0,1,2,3],[4,5,6,7,8]), # 9 gpus
]
  ngpu = len(gpus)
  if ngpu > 0:
    if config.parallelMode == 'model-parallel':
      gpusG = [gpus[i] for i in allocs[ngpu][0]]
      gpusD = [gpus[i] for i in allocs[ngpu][1]]
      return gpusG, gpusD
    else:
      gpusG = [gpus[0]]
      gpusD = [gpus[0]]
      return gpusG, gpusD
  else:
    return ["cpu"], ["cpu"]


# Decide which device we want to run on
#device = torch.device(config.gpus[0] if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")
device = config.gpus[0] if (torch.cuda.is_available() and config.ngpu > 0) else "cpu"

# allocate different gpus for Generator and Discriminator
gpusG, gpusD = allocGpus(config.gpus)
print('gpusG', gpusG)
print('gpusD', gpusD)

#dataParallel = (device.type == 'cuda') and (config.ngpu > 1) 
#dataParallel = (config.parallelMode) and (config.ngpu > 1)
print('device ', device)
print('curr_device', torch.cuda.current_device())
print('# devices=', torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print('is_available', torch.cuda.is_available())


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 1)
    #nn.init.constant_(m.bias.data, 0)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


def mycollate(batch):
    print('batch size', len(batch), batch[0].size)
    st = time.time() 
    numpy_batch = [np.array(b) for b in batch]
    print('colalte time', time.time() - st)
    collated = torch.utils.data.dataloader.default_collate(numpy_batch)
    print('collated.size', collated.size)
    collated = transforms.ToTensor(collated)
    return collated

def scaleImages(images):
  # convert from unit8 to float32
  images = images.astype('float32')
  # scale from [0,255] to [-1,1]
  images = (images - 127.5) / 127.5
  return images

def initModels():
  
 
  # Initialize loss fuconfig.nction
  #criterion = nn.MSELoss()
  criterion = nn.BCELoss()

  if config.startResLevel == 0: # if starting from scratch, then create the gen/discrim, else load from prev checkpoint
    # Create the generator and discriminator
    netG = network.Generator()
    netD = network.Discriminator()

    if config.parallelMode == 'gpu-single':
      netG.set(device)
      netD.set(device)
    if config.parallelMode == 'data-parallel':
      netG.set(device)
      netD.set(device)
  
      netG = network.MyDataParallel(netG, list(range(config.ngpu))) # allows fall-though of custom attributes
      netD = network.MyDataParallel(netD, list(range(config.ngpu)))
    elif config.parallelMode == 'model-parallel':
      netG = network.ModelParallel(netG, config.split_size, gpusG)
      netD = network.ModelParallel(netD, config.split_size, gpusD)
      

    # Print the model
    print(netG)
    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print('netG_params', netG_params)
    print(netD)
    netD_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print('netD_params', netD_params)

  else:
    #torch.save({
    #'G_state_dict' : netG.state_dict(),
    #'D_state_dict' : netD.state_dict(),
    #'G_losses' : G_losses,
    #'D_losses': D_losses,
    #'img_list' : img_list,
    #}, config.modelSavePaths[i])

    # create initial 4x4 resolution network, level = 0
    netG = network.Generator() 
    netD = network.Discriminator()

    # grow networks to resolution of previous checkpoint (8x8 - level 1, 16x16 - level 2, ...)
    netG.grow_X_levels(extraLevels=config.startResLevel-1)
    netD.grow_X_levels(extraLevels=config.startResLevel-1)

    # load weights from previous checkpoint 
    print('loading model from %s' % config.modelSavePaths[config.startResLevel - 1])
    checkpoint = torch.load(config.modelSavePaths[config.startResLevel-1])
    print('G_state', checkpoint['G_state'].keys())
    print('D_state', checkpoint['D_state'].keys())
    netG.load_state_dict(checkpoint['G_state'])
    netD.load_state_dict(checkpoint['D_state'])

    # set up correct weights for batchNorm and dropout
    netG.eval() 
    netD.eval()

    if config.parallelMode == 'gpu-single':
      netG.set(device)
      netD.set(device)
    if config.parallelMode == 'data-parallel':
      netG.set(device)
      netD.set(device)
  
      netG = network.MyDataParallel(netG, list(range(config.ngpu))) # allows fall-though of custom attributes
      netD = network.MyDataParallel(netD, list(range(config.ngpu)))
    elif config.parallelMode == 'model-parallel':
      netG = network.ModelParallel(netG, config.split_size, gpusG)
      netD = network.ModelParallel(netD, config.split_size, gpusD)


    #netG.to(device)
    #netD.to(device)
    # Handle multi-gpu if desired
    #if dataParallel:
    #  netG = network.MyDataParallel(netG, list(range(config.ngpu)))
    #  netD = network.MyDataParallel(netD, list(range(config.ngpu)))

    #netG = checkpoint['netG']
    #netD = checkpoint['netD']

  return netG, netD, criterion  

def loadBatches(l):
  curX = config.posResX[l]
  curY = config.posResY[l]
  print('Iteration ', l)
  print('Running at resolution %d x %d' % (curX, curY))

  transformPIL=transforms.Compose([
    transforms.Resize(curX),
    transforms.CenterCrop(curX),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    ])

  #transform = cvtransforms.Compose([
  #  cvtransforms.Resize(curX),
  #  cvtransforms.CenterCrop(curX),
  #  cvtransforms.ToTensor(),
  #])

  if config.loadBatchesMode == 'fileSave':

    dataset = DataLoaderOptimised.PngDataset(config.train_images_list, transformPIL)
    
    # Create the DataLoaderOptimised
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batchSize[l],
      shuffle=True, num_workers=config.workers[l], pin_memory=True,
      #collate_fn=mycollate
      )

    nr_batches_to_load = int(config.nrImgsToLoad / config.batchSize[l])

    start = time.time()
    dataBatches = [0 for x in range(nr_batches_to_load)]
    for b, data in enumerate(loader, 0):
      print('loading batch %d/%d of size %d' % (b, nr_batches_to_load, config.batchSize[l]))
      if b >= nr_batches_to_load: 
        break
      dataBatches[b] = data

    print('time for loading data', time.time() - start)

    torch.save({'dataBatches':dataBatches}, config.batchFiles[l])
  if config.loadBatchesMode == 'fileLoad':
 
    start = time.time()
    dataBatches = torch.load(config.batchFiles[l])['dataBatches']
    print('time to load images from torch file: %f' % (time.time() - start ))
    #asda
  if config.loadBatchesMode == 'on-demand':
    dataset = DataLoaderOptimised.PngDataset(config.train_images_list, transformPIL)
    
    # Create the DataLoaderOptimised
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batchSize[l],
      shuffle=True, num_workers=config.workers[l], pin_memory=True,
      #collate_fn=mycollate
      )

    dataBatches = loader

  return dataBatches



def calc_gradPenalty(netD, real_data, fake_data):
  batchSize = real_data.shape[0] 
  assert len(real_data.shape) == 4
  epsB = torch.rand(batchSize, 1, 1, 1, device=gpusD[0]) # pytorch broadcasts last 3 dimensions
  #epsB = eps.expand
  diff = real_data - fake_data
  interp = real_data + epsB * diff
  
  interpVar = torch.autograd.Variable(interp, requires_grad=True) # detach node from current computational graph, otherwise grad() below won't work
  D_interp = netD(interpVar)
  
  #netD.zero_grad()
  #compute gradient with respect to input x
  #interp.requires_grad_(True)
  #D_interp.requires_grad_(True)
  #print('netD weight', netD.main[0].weight)
  #print('netD grad', netD.main[0].weight.grad)
  #print('interp', interp)
  #print('interp.grad', interp.grad)
  
  # note, this doesn;t change the grads on the netD weights
  grads = torch.autograd.grad(outputs=D_interp, inputs=interpVar, grad_outputs=torch.ones_like(D_interp), allow_unused=True, only_inputs=False, create_graph=True, retain_graph=True)[0] 

  #print('netD grad', netD.main[0].weight.grad)
  #print('interp', interp)
  #print('interp.grad', interp.grad)
  #adsa

  #print('D_interp', D_interp)
  grads = grads.view(batchSize, -1) # collapse other dimensions
  #print('grads.shape', grads)
  gradNormsB = torch.norm(grads, dim=1) # nrBatches x 1 vector
  #print('gradNormsB', gradNormsB)
  gradPenalty = config.lambdaGrad * ((gradNormsB - 1) ** 2).mean()
  #print('gradPenalty', gradPenalty)


  return gradPenalty, gradNormsB.mean().item()
   

def oneLevel(netG, netD, criterion, dataBatches, l):
  if l != 0: # grow the network in resolution 
    #if dataParallel:
    #  netG.module.grow_network()
    #  netD.module.grow_network()
    #else:
    #print(len(netG.modules))
    netG.grow_network()
    netD.grow_network()
    
    #if config.parallelMode == 'gpu-single':
    #  netG.to(device)
    #  netD.to(device)

    print(netG)
    print(netD)


  os.system('mkdir -p %s' % config.outFolder[l])
  doPlot = True
  if doPlot:
    # Plot some training images
    real_batch = next(iter(dataBatches))
    #real_batch = dataBatches[0]
    #print('real_batch ', real_batch)
    #print('batch len', len(real_batch))
    #print(real_batch.shape)
    fig = pl.figure(figsize=(12,12))
    pl.axis("off")
    pl.title("Training Images")
    #print(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu())
    #print(np.transpose(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu(),(1,2,0)))
    pl.imshow(np.transpose(vutils.make_grid(real_batch, padding=2, normalize=True, nrow=4).cpu(),(1,2,0)))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #pl.show()
    fig.savefig('%s/sampleBatch.png' % config.outFolder[l])

  # Create batch of latent vectors that we will use to visualize
  #  the progression of the generator
  nrImgToShow = 9
  nrows = 3
  fixed_noise = torch.randn(nrImgToShow, config.latDim, 1, 1, device=gpusG[0])

  # Establish convention for real and fake labels during training
  real_label = 1
  fake_label = 0

  # Setup Adam optimizers for both G and D
  optimizerD = optim.Adam(netD.parameters(), lr=config.lr_D, betas=(config.beta1, 0.99))
  optimizerG = optim.Adam(netG.parameters(), lr=config.lr_G, betas=(config.beta1, 0.99))

  # Training Loop

  # Lists to keep track of progress
  img_list = []
  G_losses = []
  D_losses = []
  gradPenaltyI = []
  gradNormAvgI = []

  iters = 0

  one = torch.tensor(1, dtype=torch.float).to(gpusD[-1])  # alloc on last gpu of D
  mone = one * -1
  
  nrBatches = len(dataBatches)
  batchesSoFar = 0
  totalBatches = config.numEpochs[l] * nrBatches
  nrBatchesToAddAlpha = int(totalBatches / 100 / 2)

  print("Starting Training Loop...")
  # For each epoch
  for epoch in range(config.numEpochs[l]):
      if epoch == (config.numEpochs[l] - 1):
        start = time.time()

      # For each batch in the DataLoaderOptimised
      #for i, data in enumerate(loader, 0):
      for i, data in enumerate(dataBatches,0):
        batchesSoFar += 1
        
        if batchesSoFar % nrBatchesToAddAlpha == 0:
          # multiply by 2, as we want newAlpha to be 1 at 50% progress
          newAlpha = 2 * (batchesSoFar / totalBatches)
          if newAlpha <= 1.0:
            netG.updateAlpha(newAlpha)
            netD.updateAlpha(newAlpha)
          else:
             # stop blending. future updates to alpha have no effect until increase in resolution
            netG.stopBlending()
            netD.stopBlending()
       
        for a in range(config.nrAccumGrads):

         
          ############################
          # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
          ###########################
          ## Train with all-real batch
          netD.zero_grad()
          # Format batch
          #print('---------------------------------')
          #start = time.time()
          real_cpu = data.to(gpusD[0])
          #print('real_cpu.size', real_cpu.size())
        
          b_size = real_cpu.size(0)
         
          # Forward pass real batch through D
          #start = time.time()
          #print(netD)
          #print(real_cpu.get_device())
          #ads
          D_real = netD(real_cpu)
          #print('netD(real)', time.time() - start)
          # Calculate loss on all-real batch errD_real = E_data(log(D(x)))
          D_real = D_real.view(-1)
          
          # Calculate gradients for D in backward pass
          D_real_mean = D_real.mean()
          #print('D_real_mean', D_real_mean.shape)
          #print('mone', mone.shape)
          D_real_meani = D_real_mean.item()
          #start = time.time()
          #D_real_mean.backward(mone)
          #print('D_real.backward()', time.time() - start)
          
          # add small loss to keep D from drifting too far away from zero
          lossD = - D_real_mean + 0.001 * (D_real_mean ** 2)     
          lossD.backward()
          
          #label = torch.full((b_size,), real_label, device=device)
          #errD_real = criterion(D_real, label)
          #errD_real.backward()          


          ## Train with all-fake batch errD_fake = E_z [log(1 - D(G(z)))]
          # Generate batch of latent vectors
          noise = torch.randn(b_size, config.latDim , 1, 1, device=gpusG[0])
          # Generate fake image batch with G
          #start = time.time()
          fake = netG(noise).to(gpusD[0])
          #print('netG(noise)', time.time() - start)
          # Classify all fake batch with D
          D_fake = netD(fake.detach()).view(-1)

          # Calculate D's loss on the all-fake batch
          D_fake_mean = D_fake.mean()
          D_fake_meani = D_fake_mean.item()
          D_fake_mean.backward(one)

          #label.fill_(fake_label)
          #errD_fake = criterion(D_fake, label)
          #errD_fake.backward()

          #print('real_cpu', real_cpu[0,:,:,:]) 
          #print('fake', fake[0,:,:,:])
          #asd

          #loss = D_fake_mean - D_real_mean + gradPenalty
          #loss.backward() # better call backwards in each of the terms separately?

          # train with gradient penalty (Wasserstein GP)
          #start = time.time()
          gradPenalty, gradNormAvg = calc_gradPenalty(netD, real_cpu.data, fake.data)
          
          #print('calc_gradPenalty()', time.time() - start)
          #start = time.time()
          gradPenalty.backward()
          #print('gradPenalty.backward()', time.time() - start)
          gradPenalty_i = gradPenalty.item()
          #gradPenalty_i = 0

          errD = - D_real_meani + D_fake_meani + gradPenalty_i
          #errD = errD_real + errD_fake
        
          # Update D
          #start = time.time()
          optimizerD.step()
          #print('optimizerD.step()', time.time() - start)

          if i % config.n_critic == 0: # only update G evey n_critic iterations
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.latDim , 1, 1, device=gpusG[0])
            # Generate fake image batch with G
            fake = netG(noise)

            #label.fill_(real_label)  # fake labels are real for generator cost
            # Siconfig.nce we just updated D, perform another forward pass of all-fake batch through D
            D_fake_2 = netD(fake.to(gpusD[0])).view(-1)
            # Calculate G's loss based on this output
            #errG = criterion(D_fake_2, label)
            # Calculate gradients for G
            D_fake_2_mean = D_fake_2.mean()
            D_fake_mean2i = D_fake_2_mean.item()
           
            errG = - D_fake_mean2i
            #start = time.time()
            D_fake_2_mean.backward(mone)
            #print('optimizerG.step()', time.time() - start)
            #label.fill_(real_label)
            #errG = criterion(D_fake_2, label)
            #errG.backward()
    

            # Update G
            optimizerG.step()

          #if i == 20:
          #  end = time.time()
          #  print('time for last epoch', end - start)
          #  asd

          # Output training stats
          if i % config.n_critic  == 0 and ( i < 20 or i % 100 == 0) :
            print('D(real)', D_real[:10], D_real.shape)
            print('D(fake)', D_fake[:10], D_fake.shape)
            print('D(fake2)', D_fake_2[:10], D_fake_2.shape)
            print('gradPenalty %.2f  gradNormAvg %f' % (gradPenalty, gradNormAvg), '   res:%d x %d' % (config.posResX[l], config.posResY[l]))
            print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f / %.4f'
              % (epoch, config.numEpochs[l], i, len(dataBatches),
                 errD, errG, D_real_meani, D_fake_meani, D_fake_mean2i))

          # Save Losses for plotting later
          G_losses.append(errG)
          D_losses.append(errD)
          gradPenaltyI.append(gradPenalty_i)
          gradNormAvgI.append(gradNormAvg)
          

          # Check how the generator is doing by saving G's output on fixed_noise
          if i % 3000 == 0:
            with torch.no_grad():
              fake = netG(fixed_noise).detach().cpu()
              #asd
              #print(fake.shape)
              # keep track of all images in list for gnerating animated gif
              img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=nrows))
              #print(img_list)
    
            fig = pl.figure(figsize=(8,8))
            pl.clf()
            pl.title("Fake Images")
            pl.imshow(np.transpose(img_list[-1],(1,2,0)))
            #fig.show()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax = pl.gca()
            fig.savefig('%s/l%d-fake-e%02d-i%06d.png' % (config.outFolder[l], l, epoch, i))
            pl.close()
 
            saveModel(netG, netD, G_losses, D_losses, img_list)
            saveLosses(G_losses, D_losses, gradPenaltyI, gradNormAvgI)


          iters += 1

  end = time.time()
  print('time for last epoch', end - start)
  

  #if dataParallel:
  #  G_state = netG.module.state_dict()
  #  D_state = netD.module.state_dict()
  #else:

  saveModel(netG, netD, G_losses, D_losses, img_list)
 
  saveLosses(G_losses, D_losses, gradPenaltyI, gradNormAvgI)
 
  #%%capture
  fig = pl.figure(figsize=(8,8))
  pl.axis("off")
  ims = [[pl.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
  print(img_list)
  print('len ims', len(ims))
  ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
  ani.save('%s/l%d-animation.gif' % (config.outFolder[l], l), writer='imagemagick', fps=1)
  pl.close()

  #HTML(ani.to_jshtml())

  # Grab a batch of real images from the DataLoaderOptimised
  #real_batch = dataBatches[0]
  plotImages(dataBatches, nrImgToShow, nrows, img_list)

def plotFake():
  fig = pl.figure(figsize=(8,8))
            pl.clf()
            pl.title("Fake Images")
            pl.imshow(np.transpose(img_list[-1],(1,2,0)))
            #fig.show()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax = pl.gca()
            fig.savefig('%s/l%d-fake-e%02d-i%06d.png' % (config.outFolder[l], l, epoch, i))
            pl.close()
 


def plotImages(dataBatches, nrImgToShow, nrows, img_list):
  real_batch = next(iter(dataBatches))

  # Plot the real images
  fig = pl.figure(figsize=(15,7.5))
  pl.subplot(1,2,1)
  pl.axis("off")
  pl.title("Real Images")
  pl.imshow(np.transpose(vutils.make_grid(real_batch[:nrImgToShow], padding=2, normalize=True, nrow=nrows).cpu(),(1,2,0)))

  # Plot the fake images from the last epoch
  pl.subplot(1,2,2)
  pl.axis("off")
  pl.title("Fake Images")
  pl.imshow(np.transpose(img_list[-1],(1,2,0)))
  #fig.show()
  fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
  ax = pl.gca()
  fig.savefig('%s/l%d-fake.png' % (config.outFolder[l], l))
  pl.close()
 

def saveLosses(G_losses, D_losses, gradPenaltyI, gradNormAvgI):
  fig = pl.figure(figsize=(10,5))
  pl.title("Generator and Discriminator Loss During Training")
  pl.plot(G_losses, label="G")
  pl.plot(D_losses, label="D")
  pl.plot(gradPenaltyI, label="grad penalty")
  pl.plot(gradNormAvgI, label="grad norm")
  pl.xlabel("iterations")
  pl.ylabel("Loss")
  pl.legend()
  fig.savefig('%s/l%d-trainingLoss.png' % (config.outFolder[l], l))
  pl.close()




def saveModel(netG, netD, G_losses, D_losses, img_list):
  G_state = netG.state_dict()
  D_state = netD.state_dict()


  torch.save({
    'G_state' : G_state,
    'D_state' : D_state,
    #'netG_' : netG,
    #'netD' : netD,
    'G_losses' : G_losses,
    'D_losses': D_losses,
    'img_list' : img_list,
  }, config.modelSavePaths[l])

  


def rerunOneLevel(dataBatches, l=config.startResLevel):
  importlib.reload(config);
  importlib.reload(network);
  netG, netD, criterion = initModels() 
  oneLevel(netG, netD, criterion, dataBatches, l)


def rerunBatches(l=config.startResLevel):

  # import train; from train import *
  # importlib.reload(train); l = config.startResLevel; dataBatches = train.rerunBatches(l)
  # rerunOneLevel(dataBatches, l)

  importlib.reload(config);
  #importlib.reload(train);
  return loadBatches(l)

if __name__ == '__main__':

  # creates simple 2x2 models, or for 4x4 and higher loads saved checkpoints
  netG, netD, criterion = initModels() 

  # curRes = current resolution     config.posRes = list of possible resolutions
  for l in range(config.startResLevel, config.nrLevels):
    os.system("printf '\033]2;%s-%s\033\\'" % (socket.gethostname(), config.outFolder[l]))
    dataBatches = loadBatches(l)
    print(len(dataBatches))
    oneLevel(netG, netD, criterion, dataBatches, l)
    #asd    


