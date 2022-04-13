import argparse
import numpy as np
from multiprocessing import Pool

import torch
from batchgenerators.utilities.file_and_folder_operations import *

from eexo_seg_module.eexo_load_model import load_model_and_checkpoint_files
from eexo_seg_module.eexo_processing_modules import preprocess_multithreaded, save_segmentation_nifti_from_softmax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", default="images_test", required=False)
    parser.add_argument('-o', "--output_folder", default="eexo_seg_results_raw", required=False, help="folder for saving predictions")

    args = parser.parse_args()
    model = "3d_fullres"
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    folds = None
    num_parts = 1
    do_tta = True
    mixed_precision = True
    mode = 'normal'
    all_in_gpu = None
    step_size = 0.5
    force_separate_z = None
    interpolation_order = 1
    interpolation_order_z = 0
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_info_path = "pretrained_model/eexo_best.model.pkl"
    model_file_path = "pretrained_model/eexo_best.model"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open("pretrained_model/eexo_training_plans.pkl", "rb") as fr:
        expected_num_modalities = pickle.load(fr)['num_modalities']

    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [join(input_folder, i) for i in all_files]
    output_files = [join(output_folder, i) for i in all_files]

    assert len(list_of_lists) == len(output_files)

    results = []

    cleaned_output_files = []
    for o in output_files:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    pool = Pool(num_threads_nifti_save)

    torch.cuda.empty_cache()
    trainer, params = load_model_and_checkpoint_files(model, model_info_path, model_file_path)

    print("starting preprocessing generator")

    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1][None])

        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax_mean, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            None, None, force_separate_z, interpolation_order_z),)
                                          ))

    print("inference done. Finish Prediction!")

    pool.close()
    pool.join()








