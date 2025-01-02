import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from natsort import natsorted
import nibabel as nib

import os,sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)

from Data_prep.data_prep2_calc_label import calc_log_HI

def load_nifti(nii_fp, get_spacing=False):
    img = nib.load(nii_fp)
    img_array = img.get_fdata()
    spacing = img.header['pixdim'][1:4]
    if get_spacing:
        return spacing
    else:
        return img_array


class BasicDataset(Dataset):
    def __init__(self, data_dir: str, scale=1.0,  zaxis_first=True):
        if type(data_dir) is str:
            data_dir = Path(data_dir)
        pdir_list = natsorted([f for f in data_dir.iterdir() if f.is_dir()])    
        images_fp = [f.joinpath("cbf_before.nii.gz") for f in pdir_list]
        masks_fp = [f.joinpath("cbf_after.nii.gz") for f in pdir_list]
        if len(images_fp) < 1:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        assert len(images_fp) == len(masks_fp), "Mask and image must be consistent."
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [i for i in range(len(images_fp))]
        self.images_fp = images_fp
        self.masks_fp = masks_fp
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self._load_data_all()

        if zaxis_first:
            self.images = np.transpose(self.images, axes=(0,3,2,1))
            self.masks = np.transpose(self.masks, axes=(0,3,2,1))
        print(self.images.shape, self.labels.shape, self.masks.shape)

    def _load_data_all(self):
        self.images = np.asarray([load_nifti(f) for f in self.images_fp])
        self.masks = np.asarray([load_nifti(f) for f in self.masks_fp])
        spacings = [load_nifti(f,get_spacing=True) for f in self.images_fp]
        self.labels = np.asarray([calc_log_HI(image, mask, spacing=spacing)>4.5 for image,mask,spacing in zip(self.images, self.masks, spacings)])

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        # define func `load_image()`
        img = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        assert img.size == mask.size, \
            f'Image and mask should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'label': torch.as_tensor(label).long().contiguous()
        }

if __name__ == "__main__":
    test1 = BasicDataset(data_dir='/root/onethingai-tmp/mtunet_dataset')
    res = test1[1]
    image1 = (res['image']>0).int()
    mask1 = (res['mask']>0).int()
    
    print("non-zero counts for image: ", image1[:].sum())
    print("non-zero counts for mask: ", mask1[:].sum())
    print("label: ", res['label'])
