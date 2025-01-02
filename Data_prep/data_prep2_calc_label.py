from pathlib import Path
import ants
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.util import montage
import skimage.morphology as morph
import skimage.measure as meas
from natsort import natsorted 

import argparse

''' Calculate Hyper perfusion index(HI):
    1. CBR ratio map: after / before
    2. Effective Hyper-perfusion region: CBF ratio > 2 and volume > 100 mm^3 (connective volumes)
    3. weighted-sum: log(sum(CBF ratio * volume) )
    4. if log10(HI) > 4.5, hyper-perfusion = 1

    See also: Fan XY, Euro. Radio., 2021.
'''

def show_cbfratio(cbf_ratio, filename=None, cmap='jet', vmin=0,vmax=1):
    m_ = montage(np.transpose(cbf_ratio, axes=(2,1,0)), grid_shape=(5,length_z//5))
    f,ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    img2show = ax.imshow(m_, cmap=cmap, vmin=vmin, vmax=vmax)    
    f.colorbar(img2show, ax=ax, shrink=0.5, pad=0.05)
    if filename is not None:
        f.savefig(filename, dpi=150)

def calc_log_HI(cbf_before_array, cbf_after_array, 
                spacing=(1,1,1), thresh_cbf_ratio=2.0, thresh_vol=100, connective_mode=2,
               return_mask=False):
    vol_voxel = np.prod(spacing)
    
    np.seterr(divide='ignore', invalid='ignore') #ignore invalid values
    
    # cbf_ratio = cbf_after_array / cbf_before_array
    cbf_ratio = np.where(cbf_before_array == 0.0, 0.0, cbf_after_array/cbf_before_array)
    cbf_ratio[np.isnan(cbf_ratio)] = 0.
    cbf_ratio[np.isinf(cbf_ratio)] = 0.
    cbf_ratio[cbf_ratio<0] = 0.
    length_z = cbf_ratio.shape[-1]
    mask = (cbf_ratio > thresh_cbf_ratio).astype(np.uint8)
    # Connected components with connectivity 2 (aka 3D 26 connectivity) 
    labs= meas.label(mask, connectivity=connective_mode,)
    reg_list = meas.regionprops(labs)
    masks_eff_region = [] # effective masks list
    eff_regions_vals = [] 
    if len(reg_list) < 1:
        log_HI = -1
    else:
        volumes_each_component = [vol_voxel*f.num_pixels for f in reg_list]
        lab_idx_sorted = natsorted(range(len(reg_list)),key=lambda x: volumes_each_component[x], reverse=True)
        for lab_idx in lab_idx_sorted:
            if volumes_each_component[lab_idx] > 100:
                eff_regions_vals.append(lab_idx+1)
            else: break
        for eff_region_val in eff_regions_vals:
            masks_eff_region.append((labs == eff_region_val).astype(np.uint8))
        eff_vol = 0.
        for mask_eff in masks_eff_region:
            eff_vol += np.sum(cbf_ratio[mask_eff>0] * vol_voxel)
        log_HI = np.log10(eff_vol)
    if return_mask:
        return log_HI, masks_eff_region 
    else:return log_HI



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--cbf_before", help="CBF image before CEA", type=str,
                        default="/root/onethingai-tmp/mtunet_dataset/subject_1/cbf_before.nii.gz", )
    parser.add_argument("-a", "--cbf_after", help="CBF image after CEA",  type=str, 
                        default="/root/onethingai-tmp/mtunet_dataset/subject_1/cbf_after.nii.gz", )
    parser.add_argument("-tv", "--thresh_volume", type=int, default=100, help="Threshold of Effective hyper perfusion volums")
    parser.add_argument("-tr", "--thresh_ratio", type=int, default=2,  help="Threshold of Effective CBF ratio")
    parser.add_argument("-sp", "--spacing", type=str, nargs=3, default=[1.875, 1.875, 4.0],  help="Spacing of ASL images")
    parser.add_argument('--run_script', action='store_true')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()
    if args.run_script:
        data_dir = Path("/root/onethingai-tmp/mtunet_dataset/")
        pdir_list = natsorted([f for f in data_dir.iterdir()])
        for pdir in pdir_list:
            cbf_before_fp = pdir.joinpath("cbf_before.nii.gz")
            cbf_after_fp = pdir.joinpath("cbf_after.nii.gz")
            cbf_before = ants.image_read(str(cbf_before_fp))
            cbf_after = ants.image_read(str(cbf_after_fp))
            cbf_before_array = cbf_before.numpy()
            cbf_after_array = cbf_after.numpy()
            # cbf_ratio = cbf_after_array / cbf_before_array
            # cbf_ratio = np.where(cbf_before_array == 0, 0,np.divide(cbf_after_array, cbf_before_array))
            # cbf_ratio[np.isnan(cbf_ratio)] = 0.
            # cbf_ratio[np.isinf(cbf_ratio)] = 0.
            # cbf_ratio[cbf_ratio<0] = 0.
            # length_z = cbf_ratio.shape[-1]
            # vmin = np.percentile(cbf_ratio[cbf_ratio>0], 10)
            # vmax = np.percentile(cbf_ratio[cbf_ratio>0], 99)
            log_HI = calc_log_HI(cbf_before_array, cbf_after_array, spacing=(cbf_before.spacing))
            print(pdir.name, log_HI)
    else: # as command line use
        print(f'''
        Use as CLI interface. Here is input parameters:
            CBF_BEFORE: {args.cbf_before},
            CBF_AFTER: {args.cbf_after},
            THRESHOLD CBF_RATIO: {args.thresh_ratio},
            THRESHOLD VOLUME: {args.thresh_volume},
            IMAGE SPACING: {args.spacing}
        returns: log10(Hyperperfusion Index), if > 4.5 then set as `Hyper-Perfusion after CEA`.
        ''')
        cbf_before_fp = args.cbf_before
        cbf_after_fp = args.cbf_after
        cbf_before = ants.image_read(cbf_before_fp)
        cbf_after = ants.image_read(cbf_after_fp)
        cbf_before_array = cbf_before.numpy()
        cbf_after_array = cbf_after.numpy()
        log_HI = calc_log_HI(cbf_before_array, cbf_after_array, spacing=args.spacing, thresh_cbf_ratio=args.thresh_ratio, thresh_vol=args.thresh_volume,)
        print(f"Log(10) Hyperperfusion Index: {log_HI:4.3f}")