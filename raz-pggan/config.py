# Root directory for dataset
#train_images_list = "xray_local_8192.csv"
#train_images_list = "xray_pngs_8192.csv"
#train_images_list = "xray_pngs_16bit_8000.csv"
train_images_list = "xray_pngs.csv" # all 360k

# list of possible resolutions
posResX = [4,8,16,32,64,128,256,512,1024]
#posRes = [64,128,256,512,1024]
posResY = posResX
nrLevels = len(posResX)


startResLevel = 1 # starting resolution level

# model save path
modelSavePaths = ['generated/l%d-model-%dx%d.pt' % (i, posResX[i], posResY[i]) for i in range(nrLevels)]

# Number of workers for dataloader, for each growth level
workers = [10,10,10,10,10,10,10,10,10]

# Batch size during training
batchSize = [1024,1024,1024,1024,1024,256,64,16,4]

# for running tests on a subset of the data (max = 369,000)
nrImgsToLoad = 40000

keepBatchesInMemory = [True, True, True, True, True, True, True, True, True]

# Number of channels in the training images. For grayscale X-ray images this is 1
nc = 1 


# number of channels at each layer of generator (first is the dimension of latent vector)
ngc = [512, 512, 512, 512, 256, 128, 64, 32, 16]

# number of channels at each layer of discriminator 
ndc = [32, 64, 128, 256, 512, 512, 512, 512, 512]

latDim = ngc[0] # dimension of latent vector

assert nrLevels == len(ngc)
assert nrLevels == len(ndc)

# Number of training epochs
numEpochs = 20

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4


