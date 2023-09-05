import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import datapoints as dp

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgsubfolder = "PNGImages"
        self.masksubfolder = "PedMasks"
        # load all image files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 
                                                        self.imgsubfolder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 
                                                         self.masksubfolder))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgsubfolder, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masksubfolder, self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colours, return tensor of unique 
        # colours
        obj_ids = torch.unique(mask)
        # first colour is background, remove
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # unsqueeze obj_ids to tensor dim (2,1,1), broadcast over colour-coded 
        # mask to generate set of binary masks 
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        
        # get bounding box co-ords for each mask
        boxes = masks_to_boxes(masks)

        # only one class (PASpersonWalking)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        # (y2 - y1) * (x2 - x1)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowds
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # wrap sample and targets into torchvision datapoints
        img = dp.Image(img)

        target = {}
        target["boxes"] = dp.BoundingBox(boxes, format="XYXY", 
                                           spatial_size=img.shape[-2:])
        target["masks"] = dp.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
