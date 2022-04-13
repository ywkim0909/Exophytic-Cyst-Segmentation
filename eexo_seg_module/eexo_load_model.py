import pickle
import importlib
import pkgutil
import torch
import os

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr

def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    with open(pkl_file, "rb") as file_read:
        info = pickle.load(file_read)
    init = info['init']
    name = info['name']
    # search_in = os.path.join(nnunet.__path__[0], "training", "network_training")
    search_in = "eexo_seg_module"
    tr = recursive_find_python_class([search_in], name, current_module="eexo_seg_module")

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer

def load_model_and_checkpoint_files(folder, model_info_path, model_file_path, folds=None):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """

    trainer = restore_model(model_info_path)
    print(trainer)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = model_file_path
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(all_best_model_files, map_location=torch.device('cpu'))]
    return trainer, all_params
