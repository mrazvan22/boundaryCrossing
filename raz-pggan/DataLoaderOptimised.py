import os
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional

class dataloader:
  def __init__(self, config):
    pass
       
  def renew(self, resl):
    self.dataset = DicomDataset(
      train_images_list=self.train_images_list, 
      image_size= self.imsize,
      )

    self.dataloader = DataLoader(
      dataset=self.dataset,
      batch_size=self.batchsize,
      shuffle=True,
      num_workers=self.num_workers
      )

from torch.utils import data
from pathlib import Path
import pandas as pd
import pydicom
from matplotlib import cm
import time

#from PIL import Image
import cv2

class PngDataset(data.Dataset):
  def __init__(self, train_images_list, image_size, transform=None):
    self.train_images_list = pd.read_csv(train_images_list, delimiter='\n', header=None).values
    print(self.train_images_list[1,0])

    self.transform = transform
    self.image_size = image_size
  
  def __getitem__(self, index):
     
    imageFile =  self.train_images_list[index,0]
    doPrint = index % 512 == 0
    if doPrint:
      start = time.time()
      print('loading %s' % (imageFile))

    #img = Image.open(imageFile)
    img = cv2.imread(imageFile)
    if doPrint:
      end = time.time()
      print('time elapsed opening:', end - start)    
      start = time.time()
    
    img = self.transform(img)
    
    if doPrint:
      end = time.time()
      print('time elapsed transform:', end - start)    

    return img


  def __len__(self):
    # Return amount of samples of your dataset.
    return len(self.train_images_list)


class DicomDataset(data.Dataset):
  def __init__(self, train_images_list, image_size, transform=None):
    self.train_images_list = pd.read_csv(train_images_list, delimiter='\n', header=None).values
    print(self.train_images_list[1,0])
    
   
    self.transform = transform
    self.image_size = image_size

  def __getitem__(self, index):
    #  Here you have to code workload to open files or to do any kind of preprocessing.This function is submitted to multiprocessing.

    start = time.time()
    imageFile =  self.train_images_list[index,0]
    print('loading %s' % (imageFile))

    ds = pydicom.read_file(imageFile) 
    end = time.time()
    print('time elapsed reading DICOM:', end - start)    

    img_array = ds.pixel_array.astype(np.float32)
    end = time.time()
    print('time elapsed + pixel array:', end - start)    
    #imgMin = np.min(img)
    #imgMax = np.max(img)
    #img = img - imgMin / (np.max(img) - imgMin)
    img_array /= np.max(img_array)
    end = time.time()
    print('time elapsed + max normalisation:', end - start)    
    
    print('resolution=', img_array.shape)
    #print('aspect_ratio=', float(img_array.shape[0])/img_array.shape[1])
    #print('img_array', img_array.shape)

    img = Image.fromarray(np.uint8(cm.gray(img_array)*255)) # RGBA - 4 channels
    end = time.time()
    print('time elapsed + to PIL:', end - start)    

    #print('PIL img.shape', img.size)
    #print('PIL channels', img.mode)

    #from matplotlib import pyplot as pl
    #pl.imshow(img, cmap='gray')
    #pl.show()
    
    #if self.transform:
      #img = self.rescale_transform(img)
    img = self.transform(img)
    
    end = time.time()
    print('time elapsed overall loading:', end - start)    
    
    #print('final tensor shape', img.size())
    return img

  def rescale_transform(self, x):
    return torch.nn.functional.interpolate(torch.from_numpy(x).view(1,1, x.shape[0], x.shape[1]), size=(self.image_size, self.image_size), mode='nearest').view(1, self.image_size, self.image_size)
    
  def __len__(self):
    # Return amount of samples of your dataset.
    return len(self.train_images_list)

  







