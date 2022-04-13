import pickle
import importlib
import pkgutil
import torch

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
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
    with open(pkl_file, "rb") as file_read:
        info = pickle.load(file_read)
    init = info['init']
    name = info['name']
    search_in = "eexo_seg_module"
    tr = recursive_find_python_class([search_in], name, current_module="eexo_seg_module")

    trainer = tr(*init)

    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer

def load_model_and_checkpoint_files(folder, model_info_path, model_file_path, folds=None):

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
