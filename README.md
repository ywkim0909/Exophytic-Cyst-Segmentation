# ExoCystSegNet

This software package (ExoCystSegNet) includes the source codes for a fully automated segmentation of kidneys and exophytic cysts in patients with autosomal dominant polycystic kidney disease (ADPKD). The [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z#citeas) was utilized to train the neural networks, and this package is for testing the pretrained neural networks to automatically segment right kidney, left kidney, and exophytic cysts. You can refer to the following paper if you would like to know what the [exophytic cysts](https://jasn.asnjournals.org/content/31/7/1640) are:

    Bae, K. T., Shi, T., Tao, C., et al. (2020). Expanded Imaging Classification of 
    Autosomal Dominant Polycystic Kidney Disease. Journal of the American Society of 
    Nephrology, 31(7), 1640-1651.

ExoCystSegNet was trained and tested using *T<sub>2*-MR images. The paper illustrating training procedure and details of training and testing datasets was submitted to the journal, and it will be updated once the paper is published.

# Installation
Prior to using ExoCystSegNet, you need to install [PyTorch](https://pytorch.org) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Refer to the corresponding website and follow the instructions to install PyTorch and nnU-Net. 

Note that ExoCystSegNet was tested on Linux (Ubuntu 16.04 LTS), and we do not guarantee if ExoCystSegNet works on other operating systems (Windows or macOS). We recommend to use virtual environment (e.g., [Anaconda](https://anaconda.org)).

# Usage
ExoCystSegNet consists of 2 main modules: 1) automated segmentation of kidneys and exophytic cysts generating region mask files (NIfTI format) and 2) Plotting label overlayed images on MR images.

## 1) Segmentation and generation of region mask files
This module uses pretrained weights stored in [pretrained_model](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/pretrained_model) and loads test images in the [images_test](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/images_test) folder. In the [images_test](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/images_test) folder, there are 3 cases with exophytic cysts and 3 cases without exophytic cysts.

    auto_eexo_seg_main.py -i [input_folder] -o [output_folder]

The `input_folder` and `output_folder` are set to [images_test](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/images_test) and [eexo_seg_results_raw](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/eexo_seg_results_raw) as a default, respectively. Specifying input and output folders is optional, so you can just put `auto_eexo_seg_main.py` if you want to test the default files.

## 2) Plotting label overlayed images
Plotting mask overlayed images module loads the original MR images and mask files and generate multiple images (slice-by-slice) overlayed in different colors with the corresponding kidneys and exophytic cysts masks (**blue**: right kidney, **red**: left kidney, **green**: exophytic cyst). The images are saved in [eexo_seg_mask_overlay](https://github.com/ywkim0909/Exophytic-Cyst-Segmentation/tree/master/eexo_seg_mask_overlay) folder.

    plot_maskoverlay_images_main.py -i [input_folder] -l [label_folder] -o [output_folder]

Specifying input, label and output folders is also optional, so you can just put `plot_maskoverlay_images_main.py` if you want to plot the images with the default files. The examples of saved images are illustrated below:

### (Example 1) Label overlayed images (case without exophytic cysts)
![case_without_exophytic_cysts](./eexo_seg_mask_overlay/ADPKDEEXO_002_111188.png)
### (Example 2) Label overlayed images (case with exophytic cysts)
![case_with_exophytic_cysts](./eexo_seg_mask_overlay/ADPKDEEXO_005_111163.png)
