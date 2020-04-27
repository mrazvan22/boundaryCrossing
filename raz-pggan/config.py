# Root directory for dataset
#train_images_list = "xray_local_8192.csv"
#train_images_list = "xray_pngs_8192.csv"
#train_images_list = "xray_pngs_16bit_8000.csv"
train_images_list = "xray_pngs.csv" # all 360k

# Number of workers for dataloader
workers = 20

# Batch size during training
batch_size = 64

# for running tests on a subset of the data (max = 369,000)
nr_imgs_to_load = 1000

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1  # it's actually RGBA currently

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4

posRes = [2,4,8,16,32,64,128,256,512,1024]

# model save path
model_save_paths = dict([(x, 'generated/%d-model.pt' % x ) for x in posRes])

