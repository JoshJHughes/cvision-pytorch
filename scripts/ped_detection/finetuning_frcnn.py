import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes

import matplotlib.pyplot as plt

from cvision.datasets.CaltechPedCOCODataset import CaltechPedCOCODataset

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import time
from tempfile import TemporaryDirectory
import os

def load_data_caltechped(
        train_transforms = None,
        test_transforms = None,
        batch_size = 1,
        shuffle = False,
        num_workers = 0, 
        ):
    """ Load Caltech Pedestrians data into DataLoader for test & train sets.

    Args:
        train_transforms (callable, optional): A function/transform that takes
            input sample and its target as entry and returns a transformed
            version.
        train_transforms (callable, optional): A function/transform that takes
            input sample and its target as entry and returns a transformed
            version.
        batch_size (int): number of samples to be collated into a training
            batch
        shuffle (bool): set true to have data reshuffled at each epoch
        num_workers (int): number of subprocesses to use for data loading, 0 ->
            data will be loaded in the main process

    Returns:
        Dict containing 'train' & 'test' set DataLoaders with params specified
        at top of function.  
    """
    TRAIN_ROOT = 'data/Caltech_COCO/train_images/'
    TRAIN_ANNFILE = 'data/Caltech_COCO/annotations/train.json'
    TEST_ROOT = 'data/Caltech_COCO/test_images/'
    TEST_ANNFILE = 'data/Caltech_COCO/annotations/test.json'

    train_dataset = CaltechPedCOCODataset(
        root = TRAIN_ROOT,
        annFile = TRAIN_ANNFILE,
        transforms = train_transforms
    )
    test_dataset = CaltechPedCOCODataset(
        root = TEST_ROOT,
        annFile = TEST_ANNFILE,
        transforms = test_transforms
    )
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

    return dataloaders, dataset_sizes

def plot_batch(images, targets):
    """ Plot single batch of images with their bounding boxes.  

    Args:
        images (tuple of tvt.Images): Tuple of variable length containing Images
        targets (tuple of dicts): Tuple of variable length containing sample 
            targets.  Dicts must contain the key "boxes".  
    """
    ann_imgs = []
    for image, target in zip(images, targets):
        ann_imgs.append(draw_bounding_boxes(image, target['boxes'], 
                                         colors='red', width=2))
    fig, axs = plt.subplots(1,len(ann_imgs))
    for i in range(len(ann_imgs)):
        axs[i].imshow(ann_imgs[i].permute(1,2,0))
    fig.show()

def get_transforms(train):
    transforms = []
    if train:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    transforms.append(v2.ToImage())
    # transforms.append(v2.Resize((600, 400)))
    transforms.append(v2.SanitizeBoundingBoxes())
    return v2.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes, box_Score_thresh=0.9):
    # load fasterRCNN model pretrained on COCO with default weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, 
                                       box_score_thresh=box_Score_thresh)
    return model, weights

def train_model(dataloaders, dataset_sizes, device, model, preprocess, 
                optimizer, scheduler, num_epochs = 10):
    since = time.time()

    # create temp dir to save model checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_pth = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_pth)
        best_acc = 0.0
        best_loss = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")

            # each epoch has training and validation phase
            for phase in ['train', 'test']:
                match phase:
                    case 'train':
                        model.train()
                    case 'test':
                        model.eval()
                
                # change this
                running_loss = 0.0
                # running_corrects = 0

                # iterate over batches
                for images, targets in dataloaders[phase]:
                    images = [preprocess(image.to(device)) for image in images]
                    targets = [{k: v.to(device) if isinstance(v, torch.Tensor)
                                else v for k, v in t.items()} for t in targets]

                    # zero gradients
                    optimizer.zero_grad()

                    # forward, tracking gradients only if in training mode
                    with torch.set_grad_enabled(phase == 'train'):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                        # backpropagation only in training mode
                        if phase == 'train':
                            losses.backward()
                            optimizer.step()

                    # calc eval stats - change these
                    running_loss += losses
                    # running_corrects += torch.sum(preds == targets.data)

                if phase == 'train':
                    scheduler.step()

                # change this
                epoch_loss = running_loss / dataset_sizes[phase]
                # epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_acc = 0.0

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # if model has best test acc, save it
                # if phase == 'test' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     torch.save(model.state_dict(), best_model_params_pth)
                # if model has best test loss, save it
                if phase == 'test' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_pth)
            
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Acc: {best_acc:4f}')
        print(f'Best val loss: {best_loss:4f}')

        # once all epochs finished, return model with best weights
        model.load_state_dict(torch.load(best_model_params_pth))
    return model

def main():
    torchvision.disable_beta_transforms_warning()

    # params
    # our dataset has two classes only - background and person
    num_classes = 2
    box_score_thresh = 0.9
    num_epochs = 5
    # optimiser
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    # lr scheduler
    step_size=3
    gamma=0.1

    # set cuda device to gpu by default
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataloaders, dataset_sizes = load_data_caltechped(
        train_transforms=get_transforms(train=True),
        test_transforms=get_transforms(train=False),
        num_workers=0, 
        shuffle=False, 
        batch_size=2)

    # visualise first batch
    # images, targets = next(iter(dataloaders['train']))
    # plot_batch(images, targets)

    # get the model using our helper function
    model, weights = get_model(num_classes, box_score_thresh)
    # get image preprocessing transforms from weights
    preprocess = weights.transforms()
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    model = train_model(dataloaders, dataset_sizes, device, model, preprocess, 
                        optimizer, lr_scheduler, num_epochs)

    pass

if __name__ == "__main__":
    main()

# TODO
# fix eval and loss
# alter num_classes & final layers