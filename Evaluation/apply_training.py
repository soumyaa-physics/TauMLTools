import os
import sys
import json
import git
import glob
from tqdm import tqdm
from shutil import rmtree
import importlib

import uproot
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "../Training/python")
from common import setup_gpu

@hydra.main(config_path='configs', config_name='apply_training')
def main(cfg: DictConfig) -> None:
    # set up paths & gpu
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    if cfg.gpu_cfg is not None:
        setup_gpu(cfg.gpu_cfg)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # load the model
    with open(to_absolute_path(f'{path_to_artifacts}/input_cfg/metric_names.json')) as f:
        metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'
    # model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()}) # workaround to load the model without loading metric functions
    model = load_model(path_to_model)

    # load baseline training cfg and update it with parsed arguments
    training_cfg = OmegaConf.load(to_absolute_path(cfg.path_to_training_cfg))
    if cfg.training_cfg_upd is not None:
        training_cfg = OmegaConf.merge(training_cfg, cfg.training_cfg_upd)
    training_cfg = OmegaConf.to_object(training_cfg)

    if cfg.checkout_train_repo: # fetch historic git commit used to run training
        with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
            train_git_commit = active_run.data.params.get('git_commit')

        # stash local changes and checkout 
        if train_git_commit is not None:
            repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
            if cfg.verbose: print(f'\n--> Stashing local changes and checking out training commit: {train_git_commit}\n')
            repo.git.stash('save', 'stored_stash')
            repo.git.checkout(train_git_commit)
        else:
            if cfg.verbose: print('\n--> Didn\'t find git commit hash in run artifacts, continuing with current repo state\n')

    # instantiate DataLoader and get generator
    DataLoader = importlib.import_module(cfg.dataloader_module)
    scaling_cfg  = to_absolute_path(cfg.scaling_cfg)
    dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
    gen_predict = dataloader.get_predict_generator()
    tau_types_names = training_cfg['Setup']['tau_types_names']
    prop_y_glob = training_cfg['Setup']['prop_y_glob']

    pathes = glob.glob(to_absolute_path(cfg.path_to_input_dir)+'/*root') if cfg.input_filename is None \
             else [to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.input_filename}.root')]

    for input_file_name in pathes:

        # output filename definition:
        output_filename = os.path.splitext(os.path.basename(input_file_name))[0]+"_pred"  if cfg.input_filename is None \
                          else cfg.output_filename
        if os.path.exists(f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{output_filename}.h5'):
            print("File exists: ", f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{output_filename}.h5')
            if not cfg.prediction_overwrite:
                continue

        # open input file
        with uproot.open(input_file_name) as f:
            n_taus = f['taus'].num_entries
            tau_indexes = f['taus'].arrays(['evt','tau_index'], library="np")
            tau_indexes = [ tau_indexes['evt'].tolist(), tau_indexes['tau_index'].tolist() ]

            if cfg.toKeepID:
                print("Adding IDs:",cfg.toKeepID)
                toKeep = ["tau_"+id_ for id_ in cfg.toKeepID]
                deeptau_ids =  f['taus'].arrays(toKeep, library="pd")
            
        if cfg.save_input_names:
            print("Tensors will be saved:",cfg.save_input_names)
            X_saveinput = [ [] for _ in  range(len(cfg.save_input_names)) ]

        # run predictions
        predictions = []
        targets = []
        global_vars = []
        if cfg.verbose: print(f'\n\n--> Processing file {input_file_name}, number of taus: {n_taus}\n')
        with tqdm(total=n_taus) as pbar:

            for gen_return in gen_predict(input_file_name):
                
                if prop_y_glob:
                    (X,y), y_glob, indexes,size = gen_return
                    y_global = np.zeros((size, y_glob.shape[1]))
                    y_global[indexes] = y_glob
                    global_vars.append(y_global)
                else:
                    (X,y), indexes,size = gen_return

                y_pred = np.zeros((size, y.shape[1]))
                y_target = np.zeros((size, y.shape[1]))

                if dataloader.input_type=="Adversarial":
                    y_pred[indexes] = model.predict(X)[0]
                else:
                    y_pred[indexes] = model.predict(X)

                y_target[indexes] = y
                
                if cfg.save_input_names:
                    assert(len(X) == len(cfg.save_input_names))
                    for i, name in enumerate(cfg.save_input_names):
                        X_save = np.full((size,) + X[i].shape[1:], -999).astype(np.float32)
                        X_save[indexes] = X[i]
                        X_saveinput[i].append(X_save)
                
                predictions.append(y_pred)
                targets.append(y_target)

                pbar.update(size)

        # concat and check for validity
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        if np.any(np.isnan(predictions)):
            raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(
                                np.count_nonzero(np.isnan(predictions)), predictions.shape))
        if np.any(predictions < 0) or np.any(predictions > 1):
            raise RuntimeError("Predictions outside [0, 1] range.")


        # store into intermediate hdf5 file
        predictions = pd.DataFrame({f'node_{tau_type}': predictions[:, int(idx)] for idx, tau_type in tau_types_names.items()})
        targets = pd.DataFrame({f'node_{tau_type}': targets[:, int(idx)] for idx, tau_type in tau_types_names.items()}, dtype=np.int64)
        predictions.to_hdf(f'{output_filename}.h5', key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
        targets.to_hdf(f'{output_filename}.h5', key='targets', mode='r+', format='fixed', complevel=1, complib='zlib')

        if prop_y_glob:
            global_vars = np.concatenate(global_vars, axis=0)
            if np.any(np.isnan(global_vars)):
                raise RuntimeError("NaN in global_vars")
            glob_names = dataloader.config["GlobalVariables"]
            global_vars = pd.DataFrame({f'{var}': global_vars[:, int(idx)] for idx, var in enumerate(glob_names)})
            global_vars.to_hdf(f'{output_filename}.h5', key='global_vars', mode='r+', format='fixed', complevel=1, complib='zlib')
            

        # store DeepTau IDs:
        if cfg.toKeepID:
            assert(deeptau_ids.shape[0] == predictions.shape[0])
            assert(deeptau_ids.shape[0] == len(tau_indexes[0]))
            assert(deeptau_ids.shape[0] == len(tau_indexes[1]))
            deeptau_ids['event'] = tau_indexes[0]
            deeptau_ids['tau_idx'] = tau_indexes[1]
            deeptau_ids.to_hdf(f'{output_filename}.h5', key='deeptauIDs', mode='r+', format='fixed', complevel=1, complib='zlib')

        if cfg.save_input_names:
            if not os.path.exists(f'{output_filename}_input'):
                os.makedirs(f'{output_filename}_input')
            assert(len(X_saveinput) == len(cfg.save_input_names))
            for i, X_tensors in enumerate(X_saveinput):
                X_saveinput[i] = np.concatenate(X_tensors, axis=0)
            print("Saving tesnsors:")
            with tqdm(total=n_taus) as pbar:
                for tau_i, (evnt, idx) in enumerate(zip(tau_indexes[0],tau_indexes[1])):
                    saved_arrays = {}
                    for i, name in enumerate(cfg.save_input_names):
                        saved_arrays[name] = X_saveinput[i][tau_i]
                    np.save(f'{output_filename}_input/tensor_{evnt}_{idx}.npy',saved_arrays)
                    pbar.update(1)

        # log to mlflow and delete intermediate file
        with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
            mlflow.log_artifact(f'{output_filename}.h5', f'predictions/{cfg.sample_alias}')
        os.remove(f'{output_filename}.h5')

        if cfg.save_input_names:
            with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
                mlflow.log_artifact(f'{output_filename}_input', f'predictions/{cfg.sample_alias}')
            rmtree(f'{output_filename}_input')

        # log mapping between prediction file and corresponding input file
        json_filemap_name = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/pred_input_filemap.json'
        json_filemap_exists = os.path.exists(json_filemap_name)
        json_open_mode = 'r+' if json_filemap_exists else 'w'
        with open(json_filemap_name, json_open_mode) as json_file:
            if json_filemap_exists: # read performance data to append additional info
                filemap_data = json.load(json_file)
            else: # create dictionary to fill with data
                filemap_data = {}
            filemap_data[os.path.abspath(f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{output_filename}.h5')] = input_file_name
            json_file.seek(0)
            json_file.write(json.dumps(filemap_data, indent=4))
            json_file.truncate()

if __name__ == '__main__':
    repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
    current_git_branch = repo.active_branch.name
    try:
        main()  
    except Exception as e:
        print(e)
    finally:
        if 'stored_stash' in repo.git.stash('list'):
            print(f'\n--> Checking out back branch: {current_git_branch}')
            repo.git.checkout(current_git_branch)
            print(f'--> Popping stashed changes\n')
            repo.git.stash('pop')
