# Root directory for dataset
train_images_list = "xray_pngs.csv" # all 360k

# list of possible resolutions
#posResX = [4,8,16,32,64,128,256,512,1024]
posResX = [4,8,16,32,64,128,256,512,1024]
posResY = posResX
#nrLevels = len(posResX)
nrLevels = 5

startResLevel = 1 # starting resolution level

#debug = True
debug = False

#Batch size during training
#batchSize = [64,64,64,64,64,64,64,16,4]
batchSize = [16,16,16,16,16,16,14,6,3]

# Number of workers for dataloader, for each growth level
workers = [10,10,10,0,0,0,0,0,0]

# for running tests on a subset of the data (max = 369,000)
nrImgsToLoad = 40000
#nrImgsToLoad = 369000

#loadBatchesFromFile = False
loadBatchesFromFile = True

batchFiles = ['generated/batches/r%d_b%d_i%d.pt' % (posResX[r], batchSize[r], nrImgsToLoad) for r in range(len(posResX))]

keepBatchesInMemory = [True, True, True, True, True, True, True, True, True]

# Number of channels in the training images. For grayscale X-ray images this is 1
nc = 1 

# number of channels at each layer of generator (first is the dimension of latent vector)
ngc = [512, 512, 512, 512, 256, 128, 64, 32, 16]
latDim = ngc[0]

# number of channels at each layer of discriminator, in reverse order (from deep layers to shallow layers) 
ndc = [512,512,512,512,512,256,128,64,32]

#assert nrLevels == len(ngc)
#assert nrLevels == len(ndc)

# Number of training epochs
numEpochs = [2,2,2,2,2,2,2,2,2]

# Learning rate for optimizers
lr_G = 0.001 # 0.003 is too high
lr_D = 0.001

leakyParam = 0.2

# Beta1 hyperparam for Adam optimizers
beta1 = 0.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1 # use 1 GPU for small batch sizes, e.g. < 256
gpus = ['cuda%d' % i for i in range(ngpu) ]

lambdaGrad = 0.2 # this is relative to the D(x) and D(G(z)) which are both between 0 and 1 

n_critic = 1

outFolder = ['generated/lev%d_l%s_lr%s_ngc%d_ndc%d_lD%d_b%d_beta%d_nc%d_i%d' % (l, lambdaGrad, lr_G, ngc[l], ndc[l], latDim, batchSize[l], beta1, n_critic, nrImgsToLoad) for l in range(nrLevels)]


# model save path
modelSavePaths = ['%s/lev%d-model-%dx%d_b%d.pt' % (outFolder[l], l, posResX[l], posResY[l], batchSize[l]) for l in range(nrLevels)]

batchNormG = False
batchNormD = False

layerNormD = True
layerNormG = True

pixelNormD = False
pixelNormG = False

equalizeLr = True

activationFinal = 'linear' # 'linear', 'tanh', 'leaky' or 'relu' 
