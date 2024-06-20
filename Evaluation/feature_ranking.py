import os
import sys
import json
import git
import glob
from tqdm import tqdm
import importlib

import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "../Training/python")
from common import setup_gpu


'''
To trun the following code, you need to have the following files:
- A trained model in the MLflow format
- A training configuration file
- A data loader module
- A scaling configuration file
- A feature map file
- A list of input files
- A list of metric names
To run the code, you can use the following command:

For v1 version - change `config_name='feature_ranking_v1'`
python feature_ranking.py path_to_mlflow=../Training/python/distautag/mlruns/ experiment_id=7 run_id=a27159734e304ea4b7f9e0042baa9e22 path_to_input_dir=/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/DisTauTag_prod2023/TauMLTools/new-ntuples-tau-pog-v4-ext1/ sample_alias=feature_importance_stau100_lsp1_ctau100mm

For v2 version - change `config_name='feature_ranking_v2'`
python feature_ranking.py path_to_mlflow=../Training/python/distautag/mlruns/ experiment_id=7 run_id=715c0806a02a4c90856684b702c3bf2c path_to_input_dir=/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/DisTauTag_prod2023/TauMLTools/new-ntuples-tau-pog-v4-ext1/ sample_alias=feature_importance_stau100_lsp1_ctau100mm
'''

@hydra.main(config_path='configs', config_name='feature_ranking_v2')
def main(cfg: DictConfig) -> None:
    # Set up paths & GPU
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    if cfg.gpu_cfg is not None:
        setup_gpu(cfg.gpu_cfg)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    cfg_dict = OmegaConf.to_object(cfg)

    # Load the model
    with open(to_absolute_path(f'{path_to_artifacts}/input_cfg/metric_names.json')) as f:
        metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'
    model = load_model(path_to_model)

    # Load baseline training cfg and update it with parsed arguments
    training_cfg = OmegaConf.load(to_absolute_path(cfg.path_to_training_cfg))
    if cfg.training_cfg_upd is not None:
        training_cfg = OmegaConf.merge(training_cfg, cfg.training_cfg_upd)
    training_cfg = OmegaConf.to_object(training_cfg)

    if cfg.checkout_train_repo:  # Fetch historic git commit used to run training
        with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
            train_git_commit = active_run.data.params.get('git_commit')

        # Stash local changes and checkout
        if train_git_commit is not None:
            repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
            if cfg.verbose: print(f'\n--> Stashing local changes and checking out training commit: {train_git_commit}\n')
            repo.git.stash('save', 'stored_stash')
            repo.git.checkout(train_git_commit)
        else:
            if cfg.verbose: print('\n--> Didn\'t find git commit hash in run artifacts, continuing with current repo state\n')

    # Instantiate DataLoader and get generator
    DataLoader = importlib.import_module(cfg.dataloader_module)
    scaling_cfg = to_absolute_path(cfg.scaling_cfg)
    dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
    gen_predict = dataloader.get_predict_generator()
    tau_types_names = training_cfg['Setup']['tau_types_names']
    prop_y_glob = training_cfg['Setup']['prop_y_glob']
    dl_config = dataloader.config
    feature_map = dl_config["input_map"]
    cell_objects = dl_config["CellObjectType"]

    # Get the list of objects to study
    objects_to_study = cfg_dict["Objects_to_study"] if 'Objects_to_study' in cfg_dict else list(cell_objects)
    print(f"Objects to study: {objects_to_study}")

    if cfg.input_filename is None:
        paths = glob.glob(to_absolute_path(cfg.path_to_input_dir) + '/*root')
    elif isinstance(cfg_dict["input_filename"], list):
        paths = [to_absolute_path(f'{cfg.path_to_input_dir}/{file}.root') for file in cfg_dict["input_filename"]]
    else:
        paths = [to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.input_filename}.root')]

    # Initialize feature importance dictionary for the objects to study
    # feature_importance_dict = {key: np.zeros(len(feature_map[key])) for key in feature_map.keys() if key in objects_to_study}
    feature_importance_dict = {key: [] for key in cell_objects if key in objects_to_study}
    if cfg_dict["separate_first_n"]:
        separate_feature_importance_dict = {key: [[] for _ in range(cfg_dict["n_to_consider"])] for key in cell_objects if key in objects_to_study}


    events_to_take = cfg_dict["events_to_consider"] if "events_to_consider" in cfg_dict else None
    events_count = 0

    for input_file_name in paths:
        # Open input file
        with uproot.open(input_file_name) as f:
            n_taus = f['taus'].num_entries

        # Run predictions and compute gradients
        all_gradients_dict = {key: [] for key in cell_objects if key in objects_to_study}
        if cfg_dict["separate_first_n"]:
            separate_all_gradients_dict = {key: [[] for _ in range(cfg_dict["n_to_consider"])] for key in cell_objects if key in objects_to_study}

        if cfg.verbose: print(f'\n\n--> Processing file {input_file_name}, number of taus: {n_taus}\n')
        with tqdm(total=n_taus) as pbar:
            for gen_return in gen_predict(input_file_name):
                if prop_y_glob:
                    (X_list, y), y_glob, indexes, size = gen_return
                else:
                    (X_list, y), indexes, size = gen_return

                # Convert all input objects to tensors
                X_tensors = tuple([tf.convert_to_tensor(np.array(X), dtype=tf.float32) for X in X_list])

                with tf.GradientTape(persistent=True) as tape:
                    for X_tensor in X_tensors:
                        tape.watch(X_tensor)
                    preds = model(X_tensors)

                # Compute gradients only for the objects specified in `objects_to_study`
                grads_list = [tape.gradient(preds, X_tensors[i]) for i, key in enumerate(cell_objects) if key in objects_to_study]

                # Store gradients for the objects to study
                for i, key in enumerate([k for k in cell_objects if k in objects_to_study]):
                    grad = grads_list[i]
                    if grad is not None:
                        grad = grad.numpy()
                        if cfg_dict["separate_first_n"]:
                            # Separate the first n pfCandidates or lostTracks
                            if len(grad.shape) > 1:
                                for j in range(min(cfg_dict["n_to_consider"], grad.shape[1])):
                                    separate_all_gradients_dict[key][j].append(grad[:, j, :])
                        else:
                            if len(grad.shape) > 1:
                                grad = grad.reshape(-1, grad.shape[-1])
                            all_gradients_dict[key].append(grad)
                    else:
                        print(f"Warning: Gradient for {key} is None")

                pbar.update(size)
                
                # break
                if events_to_take is not None:
                    events_count += size
                    if events_count >= events_to_take:
                        break

        # Compute average gradients for the objects to study
        if cfg_dict["separate_first_n"]:
            for key in separate_all_gradients_dict.keys():
                for j in range(cfg_dict["n_to_consider"]):
                    # if separate_all_gradients_dict[key][j]:
                    separate_all_gradients_dict[key][j] = np.concatenate(separate_all_gradients_dict[key][j], axis=0)
                    avg_gradients = np.mean(np.abs(separate_all_gradients_dict[key][j]), axis=0)
                    separate_feature_importance_dict[key][j].append(avg_gradients)
        else:
            for key in all_gradients_dict.keys():
                # if all_gradients_dict[key]:
                all_gradients_dict[key] = np.concatenate(all_gradients_dict[key], axis=0)
                avg_gradients = np.mean(np.abs(all_gradients_dict[key]), axis=0)
                feature_importance_dict[key].append(avg_gradients)

        # if events_to_take is not None and events_count >= events_to_take:
        break


    if cfg_dict["separate_first_n"]:
        for key in separate_feature_importance_dict.keys():
            for j in range(cfg_dict["n_to_consider"]):
                separate_feature_importance_dict[key][j] = np.mean(np.array(separate_feature_importance_dict[key][j]), axis=0)
    else:
        # Compute final average feature importance
        for key in feature_importance_dict.keys():
            feature_importance_dict[key] = np.mean(np.vstack(feature_importance_dict[key]), axis=0)


    # Map indices to feature names using the provided feature_map
    combined_feature_names = []
    combined_feature_importance = []
    if cfg_dict["separate_first_n"]:
        for key in separate_feature_importance_dict.keys():
            for j in range(cfg_dict["n_to_consider"]):
                feature_names = [name for name, index in sorted(feature_map[key].items(), key=lambda item: item[1])]
                feature_importance_mean = separate_feature_importance_dict[key][j]
                combined_feature_names.extend([f"{name}_seq_{j}" for name in feature_names])
                combined_feature_importance.extend(feature_importance_mean)
    else:
        for key in feature_importance_dict.keys():
            feature_names = [name for name, index in sorted(feature_map[key].items(), key=lambda item: item[1])]
            feature_importance_mean = feature_importance_dict[key]
            combined_feature_names.extend(feature_names)
            combined_feature_importance.extend(feature_importance_mean)


    combined_feature_importance = np.array(combined_feature_importance)

    # Sort features by importance for better visualization
    sorted_indices = np.argsort(combined_feature_importance)[::-1]
    sorted_feature_names = [combined_feature_names[i] for i in sorted_indices]
    sorted_feature_importance = combined_feature_importance[sorted_indices]

    # Plot the feature importance
    if cfg_dict["separate_first_n"]:
        plt.figure(figsize=(14, 24))
    else:
        plt.figure(figsize=(14, 10))
    plt.barh(sorted_feature_names, sorted_feature_importance, color='skyblue')
    plt.xlabel('Average Gradient Magnitude', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Feature Importance Ranking', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout(pad=3.0)
    plt.show()
    if cfg_dict["separate_first_n"]:
        name = 'feature_importance_ranking.pdf'
    else:
        name = 'feature_importance_combined.pdf'

    plt.savefig(name)
    # Save the feature importance
    np.save('feature_importance_combined.npy', sorted_feature_importance)

    # log to mlflow and delete intermediate file
    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
        mlflow.log_artifact(name)
        mlflow.log_artifact('feature_importance_combined.npy')
    os.remove(name)
    os.remove('feature_importance_combined.npy')

if __name__ == "__main__":
    main()
