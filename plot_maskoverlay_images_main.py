import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from pylab import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Input image (nii.gz) folder name", default="images_test", required=False)
    parser.add_argument("-l", '--label_folder', help="Label image (nii.gz) folder name", default="eexo_seg_results_raw", required=False)
    parser.add_argument("-o", '--output_folder', help="Folder to save mask overlay images", default="eexo_seg_mask_overlay", required=False)

    args = parser.parse_args()
    label_folder = args.label_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    strValRootDir = Path(args.label_folder)
    strOrgRootDir = Path(args.input_folder)
    strOutDir = Path(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    nCols = 8

    fig = plt.figure(figsize=(15, 25))
    nCnt = 0
    for file_name in strValRootDir.glob('*.nii.gz'):
        nCnt += 1
        plt.clf()
        print(str(file_name))
        p_label = sitk.ReadImage(str(file_name))
        pppf_label_vol = np.array(sitk.GetArrayFromImage(p_label))

        pathImgFilename = str(file_name).replace(str(strOrgRootDir), str(strValRootDir))
        p_img = sitk.ReadImage(str(strOrgRootDir / file_name.name))
        pppf_img_vol = np.array(sitk.GetArrayFromImage(p_img))

        n_num_slices = np.shape(pppf_label_vol)[2]
        nRows = np.ceil(n_num_slices / float(nCols))

        blue = [0, 0, 255]
        red = [255, 0, 0]
        green = [0, 255, 0]

        for idx_slice in range(0, np.shape(pppf_label_vol)[2]):
            fig.add_subplot(int(nRows), int(nCols), idx_slice + 1)
            ppd_img_disp = sitk.Cast(sitk.RescaleIntensity(sitk.GetImageFromArray(pppf_img_vol[:,:,idx_slice]), outputMinimum=0, outputMaximum=255), sitk.sitkUInt8)

            ppd_label_disp = pppf_label_vol[:,:,idx_slice]
            ppd_label_disp = sitk.GetImageFromArray(ppd_label_disp)

            ppd_final_img_disp = sitk.GetArrayFromImage(sitk.LabelOverlay(image=ppd_img_disp, labelImage=ppd_label_disp, opacity=0.5, backgroundValue=0, colormap=green+blue+red))

            plt.imshow(ppd_final_img_disp)
            plt.title('%d / %d' % (idx_slice + 1, n_num_slices))
            plt.axis('off')

        plt.tight_layout()
        output_filename = file_name.name[:-7]+'.png'
        plt.savefig(str(strOutDir / output_filename))
