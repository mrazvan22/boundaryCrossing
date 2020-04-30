# Root directory for dataset
#train_images_list = "xray_local_8192.csv"
#train_images_list = "xray_pngs_8192.csv"
#train_images_list = "xray_pngs_16bit_8000.csv"
train_images_list = "xray_pngs.csv" # all 360k

# list of possible resolutions
posRes = [4,8,16,32,64,128,256,512,1024]
#posRes = [64,128,256,512,1024]

# model save path
model_save_paths = dict([(x, 'generated/%d-model.pt' % x ) for x in posRes])

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# for running tests on a subset of the data (max = 369,000)
nr_imgs_to_load = 100

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 4

# Number of channels in the training images. For grayscale X-ray images this is 1
nc = 1 


# number of channels in initial layer of generator (equal to dimension of latent vector)
ngc = 64

# number of channels in final layer of discriminator 
ndc = 64

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4


