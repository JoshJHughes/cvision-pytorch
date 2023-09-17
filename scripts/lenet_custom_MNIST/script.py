import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as T

import matplotlib.pyplot as plt
from cvision.classifiers.lenet_custom import Lenet_Custom

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def visualise(imgs):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols*rows + 1):
        idx = i-1
        img, label = imgs[idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def get_transforms():
    transforms = []
    transforms.append(T.Resize(32))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def read_data(dir, n_classes):
    # convert int class label to 1hot tensor
    onehot = lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
    data_train = datasets.MNIST(
        root=dir,
        train=True,
        download=True,
        transform=get_transforms(),
        target_transform = T.Lambda(onehot),
    )
    data_test = datasets.MNIST(
        root=dir,
        train=False,
        download=True,
        transform=get_transforms(),
        target_transform = T.Lambda(onehot),
    )
    return data_train, data_test

def train_single_epoch(train_dl, model, loss_fn, optimizer, print_nth, device):
    # set model to training mode if applicable
    model.train()
    for batch, (X, y) in enumerate(train_dl):
        # move data to gpu if appropriate
        X = X.to(device)
        y = y.to(device)
        # zero gradients
        optimizer.zero_grad()
        # compute logits & loss
        preds = model(X)
        loss = loss_fn(preds, y)
        # backpropagation
        loss.backward()
        optimizer.step()
        
        # print
        if batch % print_nth == 0:
            n_batches = len(train_dl)
            print(f"Loss: {loss.item():.5f} Batch: {batch}/{n_batches}")

def eval(test_dl, model, loss_fn, device):
    # set model to eval mode if applicable
    model.eval()
    n_data = len(test_dl.dataset)
    n_batches = len(test_dl)
    test_loss, correct = 0, 0

    # disable autograd during testing to ensure no gradients computed
    with torch.no_grad():
        for X, y in test_dl:
            # move data to gpu if appropriate
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            test_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= n_batches
    correct /= n_data
    print(f"Test Error: \n \
            Accuracy: {(100*correct):>0.1f}%, \
            Avg loss: {test_loss:>8f} \n")

def main():
    # params
    dir = 'data/'
    n_classes = 10
    # training params
    hp_batch_size = 128
    hp_epochs = 20
    print_nth = 100
    # optimiser params
    hp_lr = 0.001
    hp_momentum = 0.9
    # scheduler params
    hp_step_size = 3
    hp_gamma = 0.9

    # set device to gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load training & test data & wrap with dataloader
    train_ds, test_ds = read_data(dir, n_classes)
    train_dl = DataLoader(train_ds, batch_size=hp_batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=hp_batch_size, shuffle=True)

    # get model & move to gpu
    # model = Lenet_Custom(n_classes)
    model = Net()
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=hp_lr,
        momentum=hp_momentum
    )

    # construct a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hp_step_size,
        gamma=hp_gamma
    )

    # cross-entropy loss fn for classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # train model
    for t in range(hp_epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_single_epoch(train_dl, model, loss_fn, optimizer, print_nth, device)
        eval(test_dl, model, loss_fn, device)
        lr_scheduler.step()

    print("Done!")

if __name__ == "__main__":
    main()