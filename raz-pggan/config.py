# Root directory for dataset
# csv file should be in the format of xray_pngs_dummy.csv
train_images_list = "xray_pngs.csv" # all 360k
#train_images_list = "xray_pngs_scratch.csv" # all 360k

# list of possible resolutions
#posResX = [4,8,16,32,64,128,256,512,1024]
posResX = [4,8,16,32,64,128,256,512,1024]
posResY = posResX
nrLevels = len(posResX)
#nrLevels = len(po)

# 0 - 4x4     3 - 32 x 32      5 - 128 x 128     7 - 512 x 512 
startResLevel = 4 # starting resolution level
modelLoadPath = 'generated/lev3_l0.2_lr0.001_ngc512_ndc512_lD512_beta0_nc1_i369000/lev3-model-32x32.pt'
#modelLoadPath = 'generated/r2_lev4_l0.2_lr0.001_ngc256_ndc512_lD512_beta0_nc1_i369000/model-64x64.pt'
#modelLoadPath = 'generated/r7_lev4_l10_lr0.001_ngc256_ndc512_lD512_beta0_nc5_i369000/model-64x64.pt'

#debug = True
debug = False


# choose a preset (number of gpus, batchSizes, other params, ..)
#ngpu = 1; batchSize = [16,16,16,16,16,16,16,16,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001] # 0.003 is too high
#ngpu = 1; batchSize = [128,128,128,64,32,16,8,4,3];  lr_G = [0.0015,0.0015,0.0015,0.0015,0.0015,0.0015,0.0015,0.0015,0.0015] # 0.003 is too high
#desc = 'r1'; ngpu = 8; batchSize = [512,256,128,64,64,64,64,32,32];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001] # 0.003 is too high
#desc = 'r2'; ngpu = 8; batchSize = [512,256,128,128,128,128,64,32,32];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001] # 0.003 is too high
#desc = 'r3'; ngpu = 4; batchSize = [512,256,256,256,128,64,32,32,32];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nrAccumGrads = 8 # 0.003 is too high 
#desc = 'r4'; ngpu = 4; batchSize = [512,256,128,64,16,16,16,16,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nriccumGrads = 1; driftPenaltyD = 0.01;
#desc = 'r5'; ngpu = 4; batchSize = [256,128,64,32,16,16,8,8,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nrAccumGrads = 1; driftPenaltyD = 0.1;
#desc = 'r6'; ngpu = 4; batchSize = [512,256,128,64,16,16,16,16,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nrAccumGrads = 1; driftPenaltyD = 0.001;
#desc = 'r7'; ngpu = 4; batchSize = [512,256,128,64,16,16,16,16,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nrAccumGrads = 1; driftPenaltyD = 0.001; lambdaGrad = 10; n_critic=5
desc = 'r8'; ngpu = 4; batchSize = [512,256,128,64,16,32,32,32,16];  lr_G = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]; nrAccumGrads = 1; driftPenaltyD = 0.001; lambdaGrad = 0.2; n_critic=1

# Number of workers for dataloader, for each growth level
workers = [10,10,10,10,10,10,10,10,10]
#workers = [1,1,1,1,1,1,1,1,1]

# for running tests on a subset of the data (max = 369,000)
nrImgsToLoad = 40000
#nrImgsToLoad = 369000

#loadBatchesMode = 'on-demand'  # can be 'fileLoad', 'fileSave'
loadBatchesMode = 'fileLoad'  # can be 'fileLoad', 'fileSave'

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
#numEpochs = [1,2,2,2,2,2,2,2,2,2,2]
#numEpochs = [6,6,6,6,6,6,6,6,6,6,6]
numEpochs = [4,4,4,4,4,4,4,4,4,4,4]

# Learning rate for optimizers
lr_D = lr_G

leakyParam = 0.2

# Beta1 hyperparam for Adam optimizers
beta1 = 0.0

# Number of GPUs available. Use 0 for CPU mode.
gpus = ['cuda:%d' % i for i in range(ngpu) ]
parallelMode = 'data-parallel' # can be 'gpu-single', 'model-parallel', 'data-parallel'
if ngpu == 1:
  parallelMode = 'gpu-single'
#if ngpu[startResLevel] == 0:
#  parallelMode = 'cpu'
split_size = 4 # size of micro-batch ... 

# True - model parallelism (different layers on different gpus) False - data parallelism (different batch splits on different gpus) 

#lambdaGrad = 0.2 # this is relative to the D(x) and D(G(z)) which are both between 0 and 1 
#n_critic = 1

outFolder = ['generated/%s_lev%d_l%s_lr%s_ngc%d_ndc%d_lD%d_beta%d_nc%d_i%d' % (desc, l, lambdaGrad, lr_G[l], ngc[l], ndc[l], latDim, beta1, n_critic, nrImgsToLoad) for l in range(nrLevels)]


# model save path
modelSavePaths = ['%s/model-%dx%d.pt' % (outFolder[l], posResX[l], posResY[l]) for l in range(nrLevels)]


batchNormG = False
batchNormD = False

layerNormD = True
layerNormG = True

pixelNormD = False
pixelNormG = False

equalizeLr = True

activationFinal = 'linear' # 'linear', 'tanh', 'leaky' or 'relu' 


