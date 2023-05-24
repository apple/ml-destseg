import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.data_utils import perlin_noise


class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114)
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )

            # perlin_noise implementation
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            aug_image = self.final_preprocessing(aug_image)

            image = self.final_preprocessing(image)
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"img": image, "mask": mask}
