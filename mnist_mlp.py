import torch
from torch.autograd import Variable
# import sys
# sys.path.append('./playground_pytorch')
import sys
print(sys.path)
from utee import selector
import numpy as np

torch.manual_seed(3)

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(root='/research/boundaryCrossing/mnist_dataset', train=False, download=True),
#             batch_size=1000, shuffle=True)


model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=1000, train=False, val=True)

import pickle
pickle.dump(dict(model_raw=model_raw, ds_fetcher=ds_fetcher, is_imagenet=is_imagenet, ds_val=ds_val),
            file=open('mnist_mpl.npz', 'wb'),  protocol=pickle.HIGHEST_PROTOCOL)

stateDict = pickle.load(open('mnist_mpl.npz', 'rb'))
model_raw=stateDict['model_raw']
ds_fetcher=stateDict['ds_fetcher']
is_imagenet=stateDict['is_imagenet']
ds_val=stateDict['ds_val']


# for idx, (data, target) in enumerate(ds_val):
#   data =  Variable(torch.FloatTensor(data)).cuda()
#   output = model_raw(data)
#   print('output', output.shape)
#   print('data', data.shape)
#
#   print('predClasses', torch.argmax(output,dim=1)[:10])
#   print('trueClasses', target[:10])

#ada
model = model_raw

######################


targetClass = 0
nrTargetExempImgs = 1
sourceImageInd = 1
# dataI0DD.shape = [1000, 1, 28, 28], targetI.shape [1000]
dataI0DD, targetI = next(iter(ds_val))
initImgDD = dataI0DD[sourceImageInd, 0, :, :]
initLabelTrue = targetI[sourceImageInd]
targetExemplarInd = np.where(targetI.cpu() == targetClass)[0][:nrTargetExempImgs]
targetExemplarImgs = dataI0DD[targetExemplarInd, 0, :, :]

from boundaryCrossing import MNISTGan, VisualiserBC, MNISTInfoGan

# mnistGen = MNISTGan()
mnistGen = MNISTGan()

from mnist_cnn import DiscrimTarget

outFld = 'generated/mnistGan'
modelTarget = DiscrimTarget(model, targetClass)
visualiser = VisualiserBC(mnistGen, modelTarget, targetExemplarImgs, outFld)

print(initImgDD.shape)
predInit = model(initImgDD.view(1, 1, initImgDD.shape[0], initImgDD.shape[1]).cuda())
predTargetInit = modelTarget(initImgDD.view(1, 1, initImgDD.shape[0], initImgDD.shape[1]).cuda())

print('model(initImg)', predInit)
predClassInit = torch.argmax(predInit)
print('predClass initImg', predClassInit)
print('prob class(initImg = targetClass) = ', predTargetInit)
print('trueClass initImg', initLabelTrue)


assert predClassInit == initLabelTrue



#visualiser.testGanVsReal(dataI0DD)

visualiser.visualise(initImgDD)


