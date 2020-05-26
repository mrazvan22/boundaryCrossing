# Root directory for dataset
#train_images_list = "xray_local_8192.csv"
#train_images_list = "xray_pngs_8192.csv"
#train_images_list = "xray_pngs_16bit_8000.csv"
train_images_list = "xray_pngs.csv" # all 360k

# list of possible resolutions
#posResX = [4,8,16,32,64,128,256,512,1024]
posResX = [4,8,16,32,64,128,256]
posResY = posResX
nrLevels = len(posResX)


startResLevel = 0 # starting resolution level


#Batch size during training
#batchSize = [1024,1024,1024,1024,1024,256,64,16,4]
#batchSize = [1024,1024,1024,1024,1024,256,64,16,4]
#batchSize = [2014,1024,1024,1024,1024,256,64,16,4]
batchSize = [64,64,64,64,64,64,64,16,4]



# Number of workers for dataloader, for each growth level
workers = [10,10,10,0,0,0,0,0,0]


# for running tests on a subset of the data (max = 369,000)
#nrImgsToLoad = 40000
nrImgsToLoad = 369000

loadBatchesFromFile = True

batchFiles = ['generated/batches/r%d_%d.pt' % (r, nrImgsToLoad) for r in posResX]

keepBatchesInMemory = [True, True, True, True, True, True, True, True, True]

# Number of channels in the training images. For grayscale X-ray images this is 1
nc = 1 

batchNorm = True

layerNorm = True

pixelNorm = True # generator only

equalizeLr = True


# number of channels at each layer of generator (first is the dimension of latent vector)
#ngc = [512, 512, 512, 512, 256, 128, 64, 32, 16]
ngc = [512, 512, 512, 512, 256, 128, 64, 32, 16]

# number of channels at each layer of discriminator, in reverse order (from deep layers to shallow layers) 
#ndc = [512,512,512,512,512,256,128,64,32] 
ndc = [512,512,512,512,512,256,128,64,32]
#ndc = [32, 64, 128, 256, 512, 512, 512, 512, 512]
#ndc = [32, 64, 128, 256, 512, 512, 3, 3, 3]

ngf = 10
ndf = 10

latDim = ngc[0] # dimension of latent vector
#latDim = 20

#assert nrLevels == len(ngc)
#assert nrLevels == len(ndc)

# Number of training epochs
numEpochs = [10,10,10,10,10,10,10,10,10]

# Learning rate for optimizers
lr_G = 0.001
lr_D = 0.001

leakyParam = 0.2

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1 # use 1 GPU for small batch sizes, e.g. < 256

lambdaGrad = 0.1

n_critic = 5

#outFolder = ['generated/l%s_lr%s_ngf%d_ndf%d_lD%d_b%d_beta%s' % (lambdaGrad, lr_G, ngf, ndf, latDim, batchSize[l], beta1) for l in range(nrLevels)]
outFolder = ['generated/lev%d_l%s_lr%s_ngc%d_ndc%d_lD%d_b%d_beta%d' % (l, lambdaGrad, lr_G, ngc[l], ndc[l], latDim, batchSize[l], beta1) for l in range(nrLevels)]


# model save path
modelSavePaths = ['%s/lev%d-model-%dx%d_b%d.pt' % (outFolder[l], l, posResX[l], posResY[l], batchSize[l]) for l in range(nrLevels)]
