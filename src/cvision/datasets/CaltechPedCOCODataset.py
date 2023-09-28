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
        # this method assumes no missing data
        targetsv2 = {k: [target[k] for target in targets] for k in targets[0]}

        # add "boxes" and "labels" to store bbox and category_id in datapoints
        # required for compatibility with transforms.v2
        # note that "bbox" will not be transformed
        bboxes = torch.tensor(targetsv2['bbox'])
        targetsv2['boxes'] = dp.BoundingBox(
                torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy'),
                format = torchvision.datapoints.BoundingBoxFormat.XYXY,
                spatial_size=image.shape[-2:]
            )
        targetsv2['labels'] = torch.tensor(targetsv2['category_id'])

        if self.transforms is not None:
            image, targetsv2 = self.transforms(image, targetsv2)

        return image, targetsv2

def test():
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    from torchvision.utils import draw_bounding_boxes

    torchvision.disable_beta_transforms_warning()

    idx = 50

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

if __name__ == '__main__':
    test()
