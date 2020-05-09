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
from cvtorchvision import cvtransforms
import importlib
import pickle

#from config import *
import config
import DataLoaderOptimised
import network



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
sys.stdout.flush()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")
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

def initModels():
  
 
  # Initialize loss fuconfig.nction
  #criterion = nn.MSELoss()
  criterion = nn.BCELoss()

  if config.startResLevel == 0: # if starting from scratch, then create the gen/discrim, else load from prev checkpoint
    # Create the generator
    #netG = network.Generator(config.ngpu).to(device)
    netG = network.GeneratorDCGAN2(config.ngpu).to(device)
  

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config.ngpu > 1):
      netG = nn.DataParallel(netG, list(range(config.ngpu)))

    # Apply the weights_init fuconfig.nction to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    #netD = network.Discriminator(config.ngpu).to(device)
    netD = network.DiscriminatorDCGAN2(config.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config.ngpu > 1):
      netD = nn.DataParallel(netD, list(range(config.ngpu)))

    # Apply the weights_init fuconfig.nction to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

  else:
    #torch.save({
    #'G_state_dict' : netG.state_dict(),
    #'D_state_dict' : netD.state_dict(),
    #'G_losses' : G_losses,
    #'D_losses': D_losses,
    #'img_list' : img_list,
    #}, config.modelSavePaths[i])

    checkpoint = torch.load(config.modelSavePaths[config.startResLevel-1])
    #netG = Generator(config.ngpu, level=l-1)
    #netD = Discriminator(config.ngpu, level=l-1)
    #netG.load_state_dict(checkpoint['G_state_dict'])
    #netD.load_state_dict(checkpoint['D_state_dict'])
    #net      
    netG = checkpoint['netG']
    netD = checkpoint['netD']

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
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

  #transform = cvtransforms.Compose([
  #  cvtransforms.Resize(curX),
  #  cvtransforms.CenterCrop(curX),
  #  cvtransforms.ToTensor(),
  #])

  if not config.loadBatchesFromFile:

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
  else:  
 
    start = time.time()
    dataBatches = torch.load(config.batchFiles[l])['dataBatches']
    print('time to load images from torch file: %f' % (time.time() - start ))
    asda

  return dataBatches

def oneLevel(netG, netD, criterion, dataBatches, l):
  if l != 0: # grow the network in resolution 
    netG.module.grow_network()
    netG.to(device)
    print(netG)
    netD.module.grow_network()
    netD.to(device)
    print(netD)


  doPlot = False
  if doPlot:
    # Plot some training images
    real_batch = next(iter(loader))
    print('real_batch ', real_batch)
    print('batch len', len(real_batch))
    print(real_batch.shape)
    fig = pl.figure(figsize=(12,4))
    pl.axis("off")
    pl.title("Training Images")
    print(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu())
    print(np.transpose(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu(),(1,2,0)))
    pl.imshow(np.transpose(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu(),(1,2,0)))
    #pl.show()
    os.system('mkdir -p generated')
    fig.savefig('generated/sampleBatch.png')

  # Create batch of latent vectors that we will use to visualize
  #  the progression of the generator
  nrImgToShow = 9
  nrows = 3
  fixed_noise = torch.randn(nrImgToShow, config.latDim, 1, 1, device=device)

  # Establish convention for real and fake labels during training
  real_label = 1
  fake_label = 0

  # Setup Adam optimizers for both G and D
  optimizerD = optim.Adam(netD.parameters(), lr=config.lr_D, betas=(config.beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=config.lr_G, betas=(config.beta1, 0.999))

  # Training Loop

  # Lists to keep track of progress
  img_list = []
  G_losses = []
  D_losses = []
  iters = 0


  print("Starting Training Loop...")
  # For each epoch
  for epoch in range(config.numEpochs):
      if epoch == (config.numEpochs - 1):
        start = time.time()

      # For each batch in the DataLoaderOptimised
      #for i, data in enumerate(loader, 0):
      for i, data in enumerate(dataBatches,0):

          ############################
          # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
          ###########################
          ## Train with all-real batch
          netD.zero_grad()
          # Format batch
          real_cpu = data.to(device)
          #print('real_cpu.size', real_cpu.size())
        
          b_size = real_cpu.size(0)
          label = torch.full((b_size,), real_label, device=device)
          # Forward pass real batch through D
          output_real = netD(real_cpu)
          #print('output.size', output.size())

          # Calculate loss on all-real batch errD_real = E_data(log(D(x)))
          #print('output.shape', output.shape)
          #print('label.shape', label.shape)
          output_real = output_real.view(-1)
          errD_real = criterion(output_real, label)
          # Calculate gradients for D in backward pass
          errD_real.backward()
          D_x = output_real.mean().item()

          ## Train with all-fake batch errD_fake = E_z [log(1 - D(G(z)))]
          # Generate batch of latent vectors
          noise = torch.randn(b_size, config.latDim , 1, 1, device=device)
          # Generate fake image batch with G
          fake = netG(noise)
          #print('fake.size()', fake.size())
          label.fill_(fake_label)
          # Classify all fake batch with D
          output_fake = netD(fake.detach()).view(-1)

          # Calculate D's loss on the all-fake batch
          errD_fake = criterion(output_fake, label)
          # Calculate the gradients for this batch
          errD_fake.backward()
          D_G_z1 = output_fake.mean().item()
          # Add the gradients from the all-real and all-fake batches
          errD = errD_real + errD_fake
          # Update D
          optimizerD.step()

          ############################
          # (2) Update G network: maximize log(D(G(z)))
          ###########################
          netG.zero_grad()
          label.fill_(real_label)  # fake labels are real for generator cost
          # Siconfig.nce we just updated D, perform another forward pass of all-fake batch through D
          output_fake_2 = netD(fake).view(-1)
          # Calculate G's loss based on this output
          errG = criterion(output_fake_2, label)
          # Calculate gradients for G
          errG.backward()
          D_G_z2 = output_fake_2.mean().item()
          # Update G
          optimizerG.step()
      

          # Output training stats
          if i < 5 or i % 50 == 0:
            print('D(real)', output_real[:10], output_real.shape)
            print('D(fake)', output_fake[:10], output_fake.shape)
            print('D(fake2)', output_fake_2[:10], output_fake_2.shape)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, config.numEpochs, i, len(dataBatches),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

          # Save Losses for plotting later
          G_losses.append(errG.item())
          D_losses.append(errD.item())

          # Check how the generator is doing by saving G's output on fixed_noise
          if (i == len(dataBatches)-1):
            with torch.no_grad():
              fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=nrows))
    
            fig = pl.figure(figsize=(8,8))
            pl.title("Fake Images")
            pl.imshow(np.transpose(img_list[-1],(1,2,0)))
            #fig.show()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax = pl.gca()
            fig.savefig('generated/l%d-fake-e%02d-i%02d.png' % (l, epoch, i))
            pl.close()
 

          iters += 1

  end = time.time()
  print('time for last epoch', end - start)
  
  torch.save({
    #'G_state_dict' : netG.state_dict(),
    #'D_state_dict' : netD.state_dict(),
    'netG' : netG,
    'netD' : netD,
    'G_losses' : G_losses,
    'D_losses': D_losses,
    'img_list' : img_list,
  }, config.modelSavePaths[l])

  fig = pl.figure(figsize=(10,5))
  pl.title("Generator and Discriminator Loss During Training")
  pl.plot(G_losses,label="G")
  pl.plot(D_losses,label="D")
  pl.xlabel("iterations")
  pl.ylabel("Loss")
  pl.legend()
  os.system('mkdir -p generated')
  fig.savefig('generated/l%d-trainingLoss.png' % l)
  pl.close()

  #%%capture
  fig = pl.figure(figsize=(8,8))
  pl.axis("off")
  ims = [[pl.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
  print(img_list)
  print('len ims', len(ims))
  ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
  ani.save('generated/l%d-animation.gif' % l, writer='imagemagick', fps=1)
  pl.close()

  #HTML(ani.to_jshtml())

  # Grab a batch of real images from the DataLoaderOptimised
  real_batch = dataBatches[0]

  # Plot the real images
  fig = pl.figure(figsize=(15,7.5))
  pl.subplot(1,2,1)
  pl.axis("off")
  pl.title("Real Images")
  pl.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:nrImgToShow], padding=2, normalize=True, nrow=nrows).cpu(),(1,2,0)))

  # Plot the fake images from the last epoch
  pl.subplot(1,2,2)
  pl.axis("off")
  pl.title("Fake Images")
  pl.imshow(np.transpose(img_list[-1],(1,2,0)))
  #fig.show()
  fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
  ax = pl.gca()
  fig.savefig('generated/l%d-fake.png' % l)
  pl.close()
 
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
    dataBatches = loadBatches(l)
    print(len(dataBatches))
    #oneLevel(netG, netD, criterion, dataBatches, l)
    #asd


