import os, sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
new_env = os.environ.copy()
new_env['PATH'] += ':/root/antsInstall/install/bin/'

import argparse
from pathlib import Path
import ants
from natsort import natsorted
import numpy as np
import subprocess
from antspynet.utilities import brain_extraction, deep_atropos
from utils.logTool import logTool

ANTS_TEMP_PATH = Path("/root/onethingai-tmp/ANTs_Templates")
MNI_2mm_brain_fp = ANTS_TEMP_PATH.joinpath("MNITemplate/inst/extdata/MNI152_T1_2mm_Brain.nii.gz")
T_template0_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz")
T_template0_mask_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz")
T_temp0_regMask_fp = ANTS_TEMP_PATH.joinpath("MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz")
ANTSROOT="/root/antsInstall/install/bin/"
DATAROOT = Path('/root/onethingai-tmp/SynASL_Dataset_New/')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aslm0", help="ASL M0 image path(*.nii/*.nii.gz). ", type=str,
                        default="/root/onethingai-tmp/test_subject/sASLm0_before_25914.nii", )
    parser.add_argument("--asl", action='append',
                        help="ASL images path(*.nii/*.nii.gz) list.(PWI,CBF,ATT,...,etc) ", 
                        default=["/root/onethingai-tmp/test_subject/sASLcbf_before_25914.nii"], )
    parser.add_argument("--t1", type=str,
                        default="/root/onethingai-tmp/test_subject/T1w_before_25914.nii", 
                        help=''' T1w image path(*.nii/*.nii.gz) . ''')
    parser.add_argument("--t1_prep", choices=['b', 'w', 's'], default=[], action='append',
                        help='''Perform T1w image processing:
                                `b` for Brain Extraction. 
                                `w` for Brain Normalization.
                                `s` for Brain Tissue Segmentation. 
                                For multiple steps, e.g, you can use --t1 b --t1 s for `b` and `s`.''')
    parser.add_argument("--asl_prep",choices=['b', 'a', 'w', 'r'], default=[], action='append',
                        help='''Perform ASL processing: 
                                `b` for Brain Extraction. 
                                'a' for affine registered to T1w image.
                                `w` for Normalized to MNI coordinates. `w` contains `a`.
                                'r' for riged registered to another ASL image. 
                                For multiple steps, e.g, you can use --asl b --asl w for `b` and `w`.''')
    parser.add_argument("-o", "--out_prefix", type=str,
                        default="prep", help='''Define output prefix string. Default is `prop`.''')
    parser.add_argument("-u", "--use_exist", action="store_true", 
                        help=''' Use existsing transform for priority. Default is True. ''')
    parser.add_argument("-n", "--use_antspynet", action='store_true', 
                        help=''' Use ANTsPyNet. Default is True. if ignored, `ANTs` and `ANTsPyx` are used as Default.''')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    parser.add_argument("-l", "--save_log", action="store_true",
                        help='''
                        Save the log to show key information via terminal. 
                        will save into `log_YYMM_HHMMSS.log` in current working directory. ''')
    return parser.parse_args()

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
        fwdtransforms = reg_res['fwdtransforms']
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

if __name__ == "__main__":
    '''
    Usage:
    python __file__.py --t1_prep w --asl_prep w -u -n -l -o prep --aslm0 `ASL_M0_PATH` --asl `ASL_IMG_PATH1` --t1 `T1w_PATH`
    '''
    
    args = get_args()
    if args.save_log:
        logger = logTool(use_file=True)
    else:
        logger = logTool(use_file=False)
        
    logger.info(f''' Input Arguments:
--asl: {args.asl}, 
--aslm0: {args.aslm0},
--t1: {args.t1},
--t1_prep: {args.t1_prep},
--asl_prep: {args.asl_prep},
-u, --use_exist: {args.use_exist},
-n, --use_antspynet: {args.use_antspynet},
-o, --out_prefix: {args.out_prefix},
-l, --save_log: {args.save_log},
-v, --verbose: {args.verbose} 
''',)
    #TODO
    logger.info("Check input validation:True")
    
    
    MNI_2mm_brain_img = ants.image_read(str(MNI_2mm_brain_fp))
    t1w = ants.image_read(args.t1)
    aslm0 = ants.image_read(str(args.aslm0))
    asl_others = [ants.image_read(f) for f in args.asl]
    logger.info("Load T1w and ASL images.")
    pdir = Path(args.t1).parent
    out_prefix = args.out_prefix
    
    # T1w - BET
    t1w_mask_fp = pdir.joinpath(out_prefix+"_T1w_mask.nii.gz")
    if 'b' in args.t1 or 'w' in args.t1:
        logger.info(f"T1_prep: Brain Extraction (BET).")
        if args.use_exist:
            if t1w_mask_fp.is_file(): 
                logger.info("T1_prep - BET: Load exited data.")
                t1w_mask = ants.image_read(str(t1w_mask_fp))
            else:
                if args.use_antspynet:
                    t1w_mask = bet_ANTsPyNet(t1w, dst_path=str(t1w_mask_fp),
                                             modality='t1',prob_thresh=0.5, ri=True)
                else:
                    t1w_mask = bet_ANTs(args.t1w, dst_path=str(t1w_mask_fp.with_name(out_prefix+"T1w_mask")), 
                                        ri=True, **betkwargs)
        else:
            if args.use_antspynet:
                t1w_mask = bet_ANTsPyNet(t1w_before, dst_path=str(t1w_mask_fp),
                                         modality='t1',prob_thresh=0.5, ri=True)
            else:
                t1w_mask = bet_ANTs(str(t1w_before_fp), dst_path=str(t1w_mask_fp.with_name(out_prefix+"T1w_mask")), 
                                    ri=True, **betkwargs)
        logger.info("T1_prep - BET: Finished.")
    
    # T1w - Normalize
    if 'w' in args.t1:
        logger.info(f"T1_prep - Normalize")
        if args.use_exist:
            Syn_T1w_reg_tx = get_img_by_name(pdir,filter_patterns=f"{out_prefix}"+r"[_][r][e][g][S][y][n][Q]*", nums=-1, ri=False)
            if len(Syn_T1w_reg_tx) == 2:
                logger.info("T1_prep - Normalize: Load existed transform.")
            else:
                Syn_T1w_reg = antRegis(fixed=MNI_2mm_brain_img, moving=t1w.mask_image(t1w_mask),
                    # type_of_transform='antsRegistrationSyNQuick[s]',
                    type_of_transform='antsRegistrationSyN[s]',
                    outprefix=out_prefix+'_regSynQ',verbose=args.verbose, )
                Syn_T1w_reg_tx = Syn_T1w_reg['fwdtransforms']
        else:
            Syn_T1w_reg = antRegis(fixed=MNI_2mm_brain_img, moving=t1w.mask_image(t1w_mask),
                    # type_of_transform='antsRegistrationSyNQuick[s]',
                    type_of_transform='antsRegistrationSyN[s]',
                    outprefix=out_prefix+'regSynQ',verbose=args.verbose, )
            Syn_T1w_reg_tx = Syn_T1w_reg['fwdtransforms']
        logger.info("T1_prep - Normalize: Finished.")
        
    # T1w - Segmentation
    if 's' in args.t1:
        #TODO
        # we need to download network weights to use antspynet modules.
        pass
    # t1w_segments = deep_atropos(t1w_before,
    #                            do_preprocessing=True, use_spatial_priors=1,verbose=False,)
    
    # ASL - aff reg to T1w
    if 'w' in args.asl_prep or 'a' in args.asl:
        logger.info(f"ASL - aff reg to T1w")
        if args.use_exist:
            aff_ASLm0_tx = get_img_by_name(pdir,filter_patterns=f"{out_prefix}"+ r"[_][a][f][f][_][m][0]" + "*.mat", nums=-1, ri=False)
            if len(aff_ASLm0_tx) < 1: # no existing transform files.
                aff_ASLm0_reg = antRegis(fixed=t1w, moving=aslm0, 
                                        type_of_transform='Affine',
                                        outprefix=out_prefix+'_aff_m0',verbose = args.verbose)  
                aff_ASLm0_tx = aff_ASLm0_reg['fwdtransforms']
        
            else:
                logger.info("ASL - aff reg to T1w: Load exited transform.")
                # pass # already load transforms
        else:
            aff_ASLm0_reg = antRegis(fixed=t1w, moving=aslm0, 
                                        type_of_transform='Affine',
                                        outprefix=out_prefix+'_aff_m0',verbose = args.verbose)  
            aff_ASLm0_tx = aff_ASLm0_reg['fwdtransforms']
        logger.info("ASL - aff reg to T1w: Finished.")
        
    # ASL - normalized parameter maps
    if 'w' in args.asl_prep:
        logger.info(f"ASL - Normalize parameter maps")
        warped_asls = []*len(asl_others)
        for asl_file, asl_img in zip(args.asl, asl_others):
            warped_img = ants.apply_transforms(fixed=MNI_2mm_brain_img, moving=asl_img, 
                                transformlist = aff_ASLm0_tx+Syn_T1w_reg_tx,
                                interpolator ='linear', verbose = args.verbose,)
            warped_asls.append(warped_img)
            warped_img.to_file(str(pdir.joinpath(out_prefix+Path(asl_file).name)))
        logger.info(f"ASL - Normalize parameter maps: Finished")

