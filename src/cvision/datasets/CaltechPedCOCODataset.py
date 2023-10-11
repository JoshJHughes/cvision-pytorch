import torch
import torchvision
from torchvision import datapoints as dp
from typing import Optional, Callable, Tuple, Any

class CaltechPedCOCODataset(torchvision.datasets.coco.CocoDetection):
    """ Caltech Pedestrians dataset.  Must be in COCO format.  
    
    Dataset can be found at https://data.caltech.edu/records/f6rph-90m20

    Code to convert the dataset may be found here 
    https://github.com/mitmul/caltech-pedestrian-dataset-converter

    Requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g, 
            ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.

    note:
        :attr:`transforms` and the combination of :attr:`transform` and
            :attr:`target_transform` are mutually exclusive.
    """
    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        targets = self._load_target(id)

        # convert image to datapoints image
        image = dp.Image(image)
        
        # parent class produces labels as list of dicts for each object in image
        # transforms.v2 expects a dict of lists
        targetsv2 = {}
        # if there are annotations in image
        if len(targets) != 0:
            targetsv2['area'] = [tgt['area'] for tgt in targets]
            targetsv2['iscrowd'] = [int(tgt['iscrowd']) for tgt in targets]
            targetsv2['bbox'] = [tgt['bbox'] for tgt in targets]
            targetsv2['category_id'] = [tgt['category_id'] for tgt in targets]
            targetsv2['id'] = [tgt['id'] for tgt in targets]
            # convert bounding boxes & category_ids to transforms.v2 format
            boxes = torchvision.ops.box_convert(
                torch.tensor(targetsv2['bbox']), 'xywh', 'xyxy')
            labels = torch.tensor(targetsv2['category_id'])
            
        else:
            boxes = torch.zeros((0,4), dtype=torch.int64)
            labels = torch.ones((0,), dtype=torch.int64)
        
        targetsv2['image_id'] = id
        targetsv2['labels'] = labels
        targetsv2['boxes'] = dp.BoundingBox(
                boxes,
                format = torchvision.datapoints.BoundingBoxFormat.XYXY,
                spatial_size=image.shape[-2:]
            )
        
        if self.transforms is not None:
            image, targetsv2 = self.transforms(image, targetsv2)

        return image, targetsv2

def testplot(idx):
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    from torchvision.utils import draw_bounding_boxes

    torchvision.disable_beta_transforms_warning()

    # create untransformed dataset
    dataset = CaltechPedCOCODataset(
        root = 'data/Caltech_COCO/test_images/', 
        annFile = 'data/Caltech_COCO/annotations/test.json',
        transform = None,
        target_transform = None,
        transforms = None,
    )
    orig_image, orig_targets = dataset[idx]

    # create transformed dataset
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBox()
    ])

    dataset2 = CaltechPedCOCODataset(
        root = 'data/Caltech_COCO/test_images/', 
        annFile = 'data/Caltech_COCO/annotations/test.json',
        transform = None,
        target_transform = None,
        transforms = transforms
    )
    trans_image, trans_targets = dataset2[idx]

    ann_orig_image = draw_bounding_boxes(orig_image, orig_targets['boxes'], 
                                         colors='red', width=2)
    ann_trans_image = draw_bounding_boxes(trans_image, trans_targets['boxes'],
                                          colors='red', width=2)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(ann_orig_image.permute(1,2,0))
    axs[1].imshow(ann_trans_image.permute(1,2,0))
    fig.show()

def testloader():
    from torchvision.transforms import v2
    from torch.utils.data import DataLoader

    torchvision.disable_beta_transforms_warning()

    # data source
    TRAIN_ROOT = 'data/Caltech_COCO/train_images/'
    TRAIN_ANNFILE = 'data/Caltech_COCO/annotations/train.json'
    TEST_ROOT = 'data/Caltech_COCO/test_images/'
    TEST_ANNFILE = 'data/Caltech_COCO/annotations/test.json'

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.SanitizeBoundingBox()
    ])
    test_transforms = v2.Compose([
        v2.SanitizeBoundingBox()
    ])

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

    no_annot_idx = 10
    batch_size = 1
    shuffle = False
    num_workers = 1

    for i in range(20):
        image, target = test_dataset[i]
        pass

def testCOCO():
    from torchvision.datasets import CocoDetection
    from torchvision.datasets import wrap_dataset_for_transforms_v2
    import matplotlib.pyplot as plt
    
    dataset = CocoDetection(
        root = 'data/COCO2017/val2017',
        annFile = 'data/COCO2017/annotations/instances_val2017.json'
    )
    dataset = wrap_dataset_for_transforms_v2(dataset)

    image, target = dataset[0]

    # obtain target format for image with no annotations
    # target: {'image_id': 25593}
    for image, target in iter(dataset):
        if target['image_id'] == 25593:
            fig, axs = plt.subplots(1,1)
            axs.imshow(image)
            fig.show()
            pass
    pass


if __name__ == '__main__':
    # testplot(50) # with annots
    testplot(10) # no annots
    # testloader()
    # testCOCO()
    pass
