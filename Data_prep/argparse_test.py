import os, sys
new_env = os.environ.copy()
new_env['PATH'] += ':/root/antsInstall/install/bin/'

from pathlib import Path
import ants
from natsort import natsorted
import numpy as np
import subprocess
from antspynet.utilities import brain_extraction, deep_atropos

ANTS_TEMP_PATH = Path("/root/onethingai-tmp/ANTs_Templates")
MNI_2mm_brain_fp = ANTS_TEMP_PATH.joinpath("MNITemplate/inst/extdata/MNI152_T1_2mm_Brain.nii.gz")
T_template0_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz")
T_template0_mask_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz")
T_temp0_regMask_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz")
ANTSROOT="/root/antsInstall/install/bin/"
DATAROOT = Path('/root/onethingai-tmp/SynASL_Dataset_New/')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Directory contains T1w and ASL Nifti data", type=str,
                    default="/root/onethingai-tmp/test_subject/", )
parser.add_argument("--t1", choices=['b', 'w', 's'], default=[], action='append',
                    help='''Perform T1w image processing:
                            `b` for Brain Extraction. 
                            `w` for Brain Normalization.
                            `s` for Brain Tissue Segmentation. 
                            For multiple steps, e.g, you can use --t1 b --t1 s for `b` and `s`.''')
parser.add_argument("--asl",choices=['b', 'a', 'w', 'r'], default=[], action='append',
                    help='''Perform ASL processing: 
                            `b` for Brain Extraction. 
                            'a' for affine registered to T1w image.
                            `w` for Normalized to MNI coordinates. `w` contains `a`.
                            'r' for riged registered to another ASL image. 
                            For multiple steps, e.g, you can use --asl b --asl w for `b` and `w`.''')
parser.add_argument("-u", "--use_exist", action="store_true", 
                    help=''' Use existsing transform for priority. Default is True. 
                             The file name filter must be defined inside *.py file, or use external 
                             JSON file''')
parser.add_argument("-n", "--use_antspynet", action='store_true', 
                    help=''' Use ANTsPyNet. Default is True. if ignored, `ANTs` and `ANTsPyx` are used as Default.
                             ''')
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Increase output verbosity")

# args = parser.parse_args(['--t1','w','--asl','w','--asl', 'r', '-u','-n'])
args = parser.parse_args()
print("--path: ", args.path)
print("--t1: ", args.t1)
print("--asl: ",args.asl)
print("-u, --use_exist: ", args.use_exist)
print("-v, --verbose: ", args.verbose)
print("-n, --use_antspynet: ", args.use_antspynet)


fileName_filters_pattern = {
        't1_before':r"[T][1][w][_][b][e][f][o][r][e]_*.nii",
        't1_after':r"[T][1][w][_][a][f][t][e][r]_*.nii",
    
        'eASLm0_before' : r"[e][A][S][L][m]*[b][e][f][o][r][e]_*.nii",
        'eASLcbf_before' : r"[e][A][S][L][c]*[b][e][f][o][r][e]_*.nii",
        'eASLatt_before' : r"[e][A][S][L][a]*[b][e][f][o][r][e]_*.nii",
    
        'eASLcbf_after' : r"[e][A][S][L][c]*[a][f][t][e][r]_*.nii",
        'eASLatt_after' : r"[e][A][S][L][a]*[a][f][t][e][r]_*.nii",
        'eASLm0_after' : r"[e][A][S][L][m]*[a][f][t][e][r]_*.nii",
    
        'sASLm0_before': r"[s][A][S][L][m]*[b][e][f][o][r][e]_*.nii",
        'sASLm0_after' : r"[s][A][S][L][m]*[a][f][t][e][r]_*.nii",
    
        'sASLcbf_before' : r"[s][A][S][L][c]*[b][e][f][o][r][e]_*.nii",
        'sASLcbf_after' : r"[s][A][S][L][c]*[a][f][t][e][r]_*.nii",

        'affine_sASL_before':r"[a][f][f][_][s][A][S][L][b][e][f]*",
        'affine_sASL_after':r"[a][f][f][_][s][A][S][L][a][f][t]*",
        'affine_eASL_before':r"[a][f][f][_][e][A][S][L][b][e][f]*",
        'warp_T1w':r"[r][e][g][S][y][n][Q]*",
        'rigid_sASL_after': r"[r][i][g][i][d][_][s][A][S][L][a][f][t]*",
}

betkwargs = {'dim':'3',
             'T_temp0':str(T_template0_fp),
             'T_temp0_mask':str(T_template0_mask_fp),
             'T_temp0_regMask':str(T_temp0_regMask_fp),
             'ANTSROOT':str(ANTSROOT),
             'env': new_env, }

def bet_ANTs(src_path, dst_path, 
             dim='3', 
             T_temp0="T_template0.nii.gz", T_temp0_mask="T_template0_BrainCerebellumProbabilityMask.nii.gz",
             T_temp0_regMask="T_template0_BrainCerebellumProbabilityMask.nii.gz",
             ANTSROOT="T_template0_BrainCerebellumRegistrationMask",
             shell=False, ri=False, env=os.environ):
    command = [ANTSROOT+"antsBrainExtraction.sh", "-d", dim, "-a", src_path,  
               "-e", T_temp0, '-m',T_temp0_mask, "-f", T_temp0_regMask,
               '-o', dst_path, ]
    print(' '.join(command))
    subprocess.call(command, shell=shell, env=env)
    if ri:
        # load bet_mask image to return image suit for pipeline
        dst_fp = Path(dst_path)
        dst_stem_prefix = dst_fp.stem[:6]
        bet_mask_fp = next(dst_fp.rglob(f"{dst_stem_prefix}*Mask*"))
        bet_mask = ants.image_read(str(bet_mask_fp))
        return bet_mask
        
def bet_ANTsPyNet(src_path, dst_path, 
                  modality='t1', prob_thresh=0.5,  
                  ri=False, **kwargs):
    if isinstance(src_path, ants.ANTsImage):
        t1w_img = src_path
    else:
        t1w_img = ants.image_read(str(src_path))
    bet_res = brain_extraction(t1w_img, modality=modality, **kwargs)
    ants.image_write(bet_res,filename=str(dst_path), ri=ri)
    if ri:
        # return image: ANTs probability brain mask image. 
        return bet_res.get_mask(low_thresh = prob_thresh) 

def antRegis(fixed, moving,
             type_of_transform='Affine',
             save_transforms=True,outprefix='reg',
             **kwargs):
    reg_res = ants.registration(fixed=fixed, moving=moving, 
                                type_of_transform=type_of_transform,
                                outprefix=outprefix,
                                **kwargs, )
    if save_transforms:
        # warpedmovout = reg_res['warpedmovout']
        fwdtransforms = reg_res['fwdtransforms']
        # ants.image_write(warpedmovout, str(pdir.joinpath(outprefix+"_warped.nii.gz")))
        aff_tx = [f for f in fwdtransforms if f.endswith('.mat')]
        warp_tx = [f for f in fwdtransforms if f.endswith('.nii.gz')]
        for i in range(len(aff_tx)):
            ants.write_transform(ants.read_transform(aff_tx[i],), str(pdir.joinpath(aff_tx[i])))
        for i in range(len(warp_tx)):
            ants.image_write(ants.image_read(warp_tx[i],), str(pdir.joinpath(warp_tx[i])))
    return reg_res
    
def get_img_by_name(pdir, filter_patterns, nums=1, ri=False):
    # return list of file names, or list of `ANTsImage` object
    if nums == 1:
        img_fp = [str(next(pdir.rglob(filter_patterns)))]
    else:
        img_fp_all = natsorted([str(f) for f in pdir.rglob(filter_patterns)])
        img_fp = img_fp_all
    if ri:
        return [ants.image_read(fp) for fp in img_fp]
    else:
        return img_fp
    
MNI_2mm_brain_img = ants.image_read(str(MNI_2mm_brain_fp))
# pdir = Path(args.path)
patient_list = natsorted([f for f in DATAROOT.iterdir() if f.is_dir()])
for pdir in patient_list:
    if pdir.joinpath("rigid_sASLcbf_after.nii").is_file():continue
    
    if pdir.name.split('_')[-1] in ['50','96','113', '77']:
        print("Missing data continue:", pdir.name)
        continue
    # if pdir.name.split('_')[-1] not in ['45']:continue
    print(pdir.name)
    try:
        t1w_before = get_img_by_name(pdir, fileName_filters_pattern['t1_before'],ri=True)[0]
        t1w_before_fp = get_img_by_name(pdir, fileName_filters_pattern['t1_before'],ri=False)[0]
       
        # eASLm0_before = get_img_by_name(pdir, fileName_filters_pattern['eASLm0_before'],ri=True)[0]
        sASLm0_before = get_img_by_name(pdir, fileName_filters_pattern['sASLm0_before'],ri=True)[0]
        sASLm0_after = get_img_by_name(pdir, fileName_filters_pattern['sASLm0_after'],ri=True)[0]
        
        # eASLcbf_before = get_img_by_name(pdir, fileName_filters_pattern['eASLcbf_before'],ri=True)[0]
        # eASLatt_before = get_img_by_name(pdir, fileName_filters_pattern['eASLatt_before'],ri=True)[0]
        
        sASLcbf_before = get_img_by_name(pdir, fileName_filters_pattern['sASLcbf_before'],ri=True)[0]
        sASLcbf_after = get_img_by_name(pdir, fileName_filters_pattern['sASLcbf_after'],ri=True)[0]
        
        aff_sASLbef_tx = get_img_by_name(pdir, fileName_filters_pattern['affine_sASL_before'],nums=-1, ri=False)
        aff_sASLaft_tx = get_img_by_name(pdir, fileName_filters_pattern['affine_sASL_after'],nums=-1, ri=False)
        aff_eASLbef_tx = get_img_by_name(pdir, fileName_filters_pattern['affine_eASL_before'],nums=-1, ri=False)
        rigid_sASLaft_tx = get_img_by_name(pdir, fileName_filters_pattern['rigid_sASL_after'],nums=-1, ri=False)
        warp_T1w_tx = get_img_by_name(pdir, fileName_filters_pattern['warp_T1w'],nums=-1, ri=False)
        
    except:
        print(f"Missing files of {pdir}")
        sys.exit()
    # T1w - BET
    t1w_mask_fp = pdir.joinpath("T1w_mask.nii.gz")
    if 'b' in args.t1 or 'w' in args.t1:
        if args.use_exist:
            if t1w_mask_fp.is_file(): 
                t1w_mask = ants.image_read(str(t1w_mask_fp))
                print("T1w - BET: Load exited data.")
            else:
                print(f" ERROR: Cannot find {str(t1w_mask_fp.name)}. Calcualtion now.")
                # sys.exit()
                if args.use_antspynet:
                    t1w_mask = bet_ANTsPyNet(t1w_before, dst_path=str(t1w_mask_fp),
                                             modality='t1',prob_thresh=0.5, ri=True)
                else:
                    t1w_mask = bet_ANTs(str(t1w_before_fp), dst_path=str(t1w_mask_fp.with_name("T1w_mask_bet")), 
                                        ri=True, **betkwargs)
        else:
            if args.use_antspynet:
                t1w_mask = bet_ANTsPyNet(t1w_before, dst_path=str(t1w_mask_fp),
                                         modality='t1',prob_thresh=0.5, ri=True)
            else:
                t1w_mask = bet_ANTs(str(t1w_before_fp), dst_path=str(t1w_mask_fp.with_name("T1w_mask_bet")), 
                                    ri=True, **betkwargs)
    
    # T1w - Normalize
    if 'w' in args.t1:
        if args.use_exist:
            if len(warp_T1w_tx) < 1: # no existing transform files.
                print(f" ERROR: Cannot find T1w - Normalize Files. Calculate now ...")
                # sys.exit()
                Syn_T1w_reg = antRegis(fixed=MNI_2mm_brain_img, moving=t1w_before.mask_image(t1w_mask),
                    # type_of_transform='antsRegistrationSyNQuick[s]',
                    type_of_transform='antsRegistrationSyN[s]',
                    outprefix='regSynQ',verbose=args.verbose, )
                Syn_T1w_reg_tx = Syn_T1w_reg['fwdtransforms']
            else:
                Syn_T1w_reg_tx = warp_T1w_tx
                print("T1w - Normalize: Load exited transform.")
        else:
            Syn_T1w_reg = antRegis(fixed=MNI_2mm_brain_img, moving=t1w_before.mask_image(t1w_mask),
                    # type_of_transform='antsRegistrationSyNQuick[s]',
                    type_of_transform='antsRegistrationSyN[s]',
                    outprefix='regSynQ',verbose=args.verbose, )
            Syn_T1w_reg_tx = Syn_T1w_reg['fwdtransforms']
    
    
    # T1w - Segmentation
    if 's' in args.t1:
        #TODO
        # we need to download network weights to use antspynet modules.
        pass
    # t1w_segments = deep_atropos(t1w_before,
    #                            do_preprocessing=True, use_spatial_priors=1,verbose=False,)
    
    # ASL - aff reg to T1w
    if 'a' in args.asl or 'w' in args.asl:
        if args.use_exist:
            if len(aff_eASLbef_tx) < 1: # no existing transform files.
                print(f"ERROR: Cannot find Files. Calculate now ...")
                aff_sASLbef_reg = antRegis(fixed=t1w_before, moving=sASLm0_before, 
                                        type_of_transform='Affine',
                                        outprefix='aff_sASLbef',verbose = args.verbose)  
                aff_sASLbef_tx = aff_sASLbef_reg['fwdtransforms']
        
                aff_sASLaft_reg = antRegis(fixed=t1w_before, moving=sASLm0_after, 
                                            type_of_transform='Affine',
                                            outprefix='aff_sASLaft',verbose = args.verbose)
                aff_sASLaft_tx = aff_sASLaft_reg['fwdtransforms']
                
                aff_eASLbef_reg = antRegis(fixed=t1w_before, moving=eASLm0_before, 
                                            type_of_transform='Affine',
                                            outprefix='aff_eASLbef',verbose = args.verbose)
                aff_eASLbef_tx = aff_eASLbef_reg['fwdtransforms']
            else:
                print("ASL - aff reg to T1w: Load exited transform.")
                # pass # already load transforms
        else:
            aff_sASLbef_reg = antRegis(fixed=t1w_before, moving=sASLm0_before, 
                                        type_of_transform='Affine',
                                        outprefix='aff_sASLbef',verbose = args.verbose)  
            aff_sASLbef_tx = aff_sASLbef_reg['fwdtransforms']
    
            aff_sASLaft_reg = antRegis(fixed=t1w_before, moving=sASLm0_after, 
                                        type_of_transform='Affine',
                                        outprefix='aff_sASLaft',verbose = args.verbose)
            aff_sASLaft_tx = aff_sASLaft_reg['fwdtransforms']
            
            aff_eASLbef_reg = antRegis(fixed=t1w_before, moving=eASLm0_before, 
                                        type_of_transform='Affine',
                                        outprefix='aff_eASLbef',verbose = args.verbose)
            aff_eASLbef_tx = aff_eASLbef_reg['fwdtransforms']
        
    # ASL - normalized parameter maps
    if 'w' in args.asl:
        warped_sASLcbf_before = ants.apply_transforms(fixed=MNI_2mm_brain_img, moving=sASLcbf_before, 
                            transformlist = aff_sASLbef_tx+Syn_T1w_reg_tx,
                            interpolator ='linear', verbose = args.verbose,)
        warped_eASLcbf_before = ants.apply_transforms(fixed=MNI_2mm_brain_img, moving=eASLcbf_before, 
                            transformlist = aff_eASLbef_tx+Syn_T1w_reg_tx,
                            interpolator ='linear', verbose = args.verbose,)
        warped_eASLatt_before = ants.apply_transforms(fixed=MNI_2mm_brain_img, moving=eASLatt_before, 
                            transformlist = aff_eASLbef_tx+Syn_T1w_reg_tx,
                            interpolator ='linear', verbose = args.verbose,)
        warped_sASLcbf_after = ants.apply_transforms(fixed=MNI_2mm_brain_img, moving=sASLcbf_after, 
                            transformlist = aff_sASLaft_tx+Syn_T1w_reg_tx,
                            interpolator ='linear', verbose = args.verbose,)
    
    
    # ASL - rigid reg to ASL#1
    if 'r' in args.asl:
        if args.use_exist:
            if len(rigid_sASLaft_tx) < 1:
                print(f" ERROR: Cannot find files. Calculate now.")
                rigid_sASLaft_reg = antRegis(fixed=sASLm0_before, moving=sASLm0_after, 
                            type_of_transform='Rigid', outprefix="rigid_sASLaft", verbose=False)
                rigid_sASLaft_tx = rigid_sASLaft_reg['fwdtransforms']
        else:
            
            rigid_sASLaft_reg = antRegis(fixed=sASLm0_before, moving=sASLm0_after, 
                            type_of_transform='Rigid', outprefix="rigid_sASLaft", verbose=False)
            rigid_sASLaft_tx = rigid_sASLaft_reg['fwdtransforms']
            
        rigid_sASLcbf_after = ants.apply_transforms(fixed=sASLm0_before, moving=sASLcbf_after, 
                            transformlist =rigid_sASLaft_tx ,
                            interpolator ='linear',
                            verbose =False,)
        ants.image_write(rigid_sASLcbf_after, str(pdir.joinpath("rigid_sASLcbf_after.nii")))
    # break
