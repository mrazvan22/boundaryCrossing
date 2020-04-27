#from __future__ import print_function
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
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from IPython.display import HTML
import time
from cvtorchvision import cvtransforms


from config import *
import DataLoaderOptimised
from network import *


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
sys.stdout.flush()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
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

def train():


  print('curr_device', torch.cuda.current_device())
  print('# devices=', torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))
  print('is_available', torch.cuda.is_available())

  
  # Decide which device we want to run on
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
  print('device ', device)

  for curentRes in posRes:

    transformPIL=transforms.Compose([
      transforms.Resize(curentRes),
      transforms.CenterCrop(curentRes),
      transforms.ToTensor(),
      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])

    transform = cvtransforms.Compose([
      cvtransforms.Resize(curentRes),
      cvtransforms.CenterCrop(curentRes),
      cvtransforms.ToTensor(),
    ])

    #randTensor = torch.randn(2544,3056)
    #print(transforms.Resize(curentRes)(randTensor).shape)
    #print(transforms.CenterCrop(curentRes)(randTensor).shape)
    #print(transforms.CenterCrop(curentRes)(randTensor).shape)
    #print(adas)

    dataset = DataLoaderOptimised.PngDataset(train_images_list, curentRes, transform)
    
    # Create the DataLoaderOptimised
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
      shuffle=True, num_workers=workers, pin_memory=True,
      #collate_fn=mycollate
      )


    doPlot = False
    if doPlot:
      # Plot some training images
      real_batch = next(iter(loader))
      print('real_batch ', real_batch)
      print('batch len', len(real_batch))
      print(real_batch.shape)
      pl.figure(figsize=(12,4))
      pl.axis("off")
      pl.title("Training Images")
      pl.imshow(np.transpose(vutils.make_grid(real_batch.to(device), padding=2, normalize=True, nrow=8).cpu(),(1,2,0)))
      pl.show()

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
      netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
      netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    nr_batches_to_load = int(nr_imgs_to_load / batch_size)

    start = time.time()
    dataBatches = [0 for x in range(nr_batches_to_load)]
    for b, data in enumerate(loader, 0):
      if b >= nr_batches_to_load: 
        break
      dataBatches[b] = data

    print('time for loading data', time.time() - start)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        if epoch == (num_epochs - 1):
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
          
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu)

            # Calculate loss on all-real batch
            #print('output.shape', output.shape)
            #print('label.shape', label.shape)
            output = output.view(-1)
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
        

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i == len(dataBatches)-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    end = time.time()
    print('time for last epoch', end - start)
    
    torch.savea({
      'state_dict' : model.state_dict(),
      'G_losses' : G_losses,
      'D_losses', D_losses,
      ''
    }, model_save_paths[cur_res])

    fig = pl.figure(figsize=(10,5))
    pl.title("Generator and Discriminator Loss During Training")
    pl.plot(G_losses,label="G")
    pl.plot(D_losses,label="D")
    pl.xlabel("iterations")
    pl.ylabel("Loss")
    pl.legend()
    fig.show()
    fig.savefig('generated/%d-trainingLoss.png')

    #%%capture
    fig = pl.figure(figsize=(8,8))
    pl.axis("off")
    ims = [[pl.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the DataLoaderOptimised
    real_batch = dataBatches[0]

    # Plot the real images
    pl.figure(figsize=(15,15))
    pl.subplot(1,2,1)
    pl.axis("off")
    pl.title("Real Images")
    pl.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    pl.subplot(1,2,2)
    pl.axis("off")
    pl.title("Fake Images")
    pl.imshow(np.transpose(img_list[-1],(1,2,0)))
    pl.show()



if __name__ == '__main__':

  train()
