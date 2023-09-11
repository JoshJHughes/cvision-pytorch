import os
import numpy as np
import torch
from PIL import Image

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
        img = Image.open(img_path).convert("RGB")
        # don't convert mask to RGB as each colour corresponds to different
        # instance, w. 0 = background
        mask = Image.open(mask_path)
        # convert PIL Image to numpy array
        mask = np.array(mask)
        # instances encoded as different colours
        obj_ids = np.unique(mask)
        # first colour is background, remove
        obj_ids = obj_ids[1:]

        # unsqueeze obj_ids to tensor dim (2,1,1), broadcast over colour-coded 
        # mask to generate set of binary masks 
        masks = (mask == obj_ids[:, None, None])
        
        # get bounding box co-ords for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # only one class (PASpersonWalking)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        image_id = idx
        # (y2 - y1) * (x2 - x1)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
