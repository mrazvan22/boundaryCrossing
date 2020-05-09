from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

from matplotlib import pyplot as pl

class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class DiscrimTarget(object):
    def __init__(self, model, targetClass):
        self.model = model
        self.targetClass = targetClass

    # return number between 0 and 1, probability of x being in class self.targetClass
    def __call__(self, x, printShape=False):
        if len(x.shape) == 2:
            xReshaped = x.view(1,1,x.shape[0], x.shape[1]).cuda()
            predTarget = self.normaliseActs(self.model(xReshaped))[:, self.targetClass]
        else:
            predTarget = self.normaliseActs(self.model(x))[:,self.targetClass]
        if printShape:
            print('predSize = ', self.model(x).shape)
            print('predTarget', predTarget)
        # print('self.normaliseActs(self.model(x))', self.normaliseActs(self.model(x)))
        # asd
        return predTarget

    # return number between 0 and 1, probability of x being in class self.targetClass
    def predMaxLikClass(self, x):
        if len(x.shape) == 2:
            xReshaped = x.view(1,1,x.shape[0], x.shape[1]).cuda()
            print('xReshaped', xReshaped.shape)
            print('self.model(xReshaped)', self.model(xReshaped).shape)
            print(self.model(xReshaped))
            print(torch.argmax(self.model(xReshaped)))
            predTarget = torch.argmax(self.model(xReshaped))
        else:
            predTarget = torch.argmax(self.model(x), dim=1)
        return predTarget

    def normaliseActs(self, activations):
        expActs = torch.exp(activations)
        return expActs / expActs.sum()

    def parameters(self):
        return self.model.parameters()



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=True,
                        help='For loading the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataI0DD', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataI0DD', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = MnistDiscriminator().to(device)

    if args.save_model:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
            scheduler.step()
        torch.save(model.state_dict(), "mnist_cnn.pt")

    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn.pt"))
        # torch.load(model.state_dict(), "mnist_cnn.pt")


    targetClass = 0
    nrTargetExempImgs = 5

    # dataI0DD.shape = [1000, 1, 28, 28], targetI.shape [1000]
    dataI0DD, targetI = next(iter(test_loader))
    initImg = dataI0DD[0,0,:,:]
    targetExemplarInd = np.where(targetI.cpu() == targetClass)[0][:nrTargetExempImgs]
    targetExemplarImgs = dataI0DD[targetExemplarInd,0,:,:]

    from boundaryCrossing import MNISTGan, VisualiserBC, MNISTInfoGan
    # mnistGen = MNISTGan()
    mnistGen = MNISTInfoGan()

    outFld = 'generated/mnistGan'
    modelTarget = DiscrimTarget(model, targetClass)
    visualiser = VisualiserBC(mnistGen, modelTarget, targetExemplarImgs, outFld)

    print(initImg.shape)
    predInit = modelTarget(initImg.view(1,1,initImg.shape[0], initImg.shape[1]).cuda())
    print('model(initImg)', predInit)
    classInit = torch.argmax(predInit)
    print('classInit', classInit)
    # asda

    visualiser.testGanVsReal(dataI0DD)

    visualiser.visualise(initImg)

    # with torch.no_grad():
    #     dataI0DD, targetI = next(iter(test_loader))
    #     dataI0DD, targetI = dataI0DD.to(device), targetI.to(device)
    #     output = model(dataI0DD)
    #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #     imgs = dataI0DD.cpu()
    #     imgs = imgs.reshape(imgs.shape[0], imgs.shape[2], imgs.shape[3])
    #
    #     R, C = 5, 5
    #     nrImg = R * C
    #     for i in range(nrImg):
    #         pl.subplot(R, C, i + 1)
    #         pl.imshow(imgs[i], cmap='gray')
    #
    #     print('pred[:nrImg]', pred[:nrImg])
    #     fig = pl.gcf()
    #     pl.show()






if __name__ == '__main__':
    main()
