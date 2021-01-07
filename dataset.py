import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
import os
import skimage
from PIL import Image
from torchvision import transforms
import random


transform_resize_for_train = transforms.Compose([
    transforms.Resize((41, 41))
])


# Convert to YCbCr format and return only the Y channel
def get_Y_channel(img):
    return np.array(img.convert('YCbCr'))[:, :, 0].astype(float)/255


class TrainDataset(Dataset):
    def __init__(self, file_path):
        self.hr_imgs = []
        self.lr_imgs = []
        for filename in os.listdir(file_path):
            tmp = Image.open(os.path.join(file_path, filename))
            tmp_hr = transform_resize_for_train(tmp)
            # Generate the low resolution version of the same image
            tmp_lr = transform_resize_for_train(
                transforms.Resize((int(tmp.size[0]/3), int(tmp.size[1]/3)))(tmp))

            self.hr_imgs.append(tmp_hr)
            self.lr_imgs.append(tmp_lr)

            # augmentation - vflip
            self.hr_imgs.append(transforms.functional.vflip(tmp_hr))
            self.lr_imgs.append(transforms.functional.vflip(tmp_lr))
            # - hflip
            self.hr_imgs.append(transforms.functional.hflip(tmp_hr))
            self.lr_imgs.append(transforms.functional.hflip(tmp_lr))

    def __getitem__(self, index):

        # augmentation - random angle rotation
        random_angle = random.randint(1, 360)

        tmp_lr = transforms.functional.rotate(
            self.lr_imgs[index], angle=random_angle)
        tmp_lr = get_Y_channel(tmp_lr)

        tmp_hr = transforms.functional.rotate(
            self.hr_imgs[index], angle=random_angle)
        tmp_hr = get_Y_channel(tmp_hr)

        return transforms.ToTensor()(tmp_lr), transforms.ToTensor()(tmp_hr)

    def __len__(self):
        return len(self.hr_imgs)
