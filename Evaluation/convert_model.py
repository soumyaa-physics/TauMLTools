import os
import sys
import json
import git
import glob
from tqdm import tqdm

import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import cmsml

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


    path_to_model = f'{path_to_artifacts}/model'
    model = load_model(path_to_model) 

    # load baseline training cfg and update it with parsed arguments
    training_cfg = OmegaConf.load(to_absolute_path(cfg.path_to_training_cfg))
    if cfg.training_cfg_upd is not None:
        training_cfg = OmegaConf.merge(training_cfg, cfg.training_cfg_upd)
    training_cfg = OmegaConf.to_object(training_cfg)

    # if cfg.checkout_train_repo: # fetch historic git commit used to run training
    #     with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
    #         train_git_commit = active_run.data.params.get('git_commit')

    #     # stash local changes and checkout 
    #     if train_git_commit is not None:
    #         repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
    #         if cfg.verbose: print(f'\n--> Stashing local changes and checking out training commit: {train_git_commit}\n')
    #         repo.git.stash('save', 'stored_stash')
    #         repo.git.checkout(train_git_commit)
    #     else:
    #         if cfg.verbose: print('\n--> Didn\'t find git commit hash in run artifacts, continuing with current repo state\n')

    #instantiate DataLoader and get generator
    import DataLoaderReco
    scaling_cfg  = to_absolute_path(cfg.scaling_cfg)
    dataloader = DataLoaderReco.DataLoader(training_cfg, scaling_cfg)

    tensor_shape, tensor_type  = dataloader.get_shape()
    input_shape = [ list(t) for t in tensor_shape[0] ]
    n_tau = 1
    input_shape_tuple = []
    for part in input_shape:
        part[0] = n_tau
        input_shape_tuple.append(tuple(part))

    print(tensor_shape, tensor_type)
    print((tf.TensorSpec(shape=input_shape_tuple[0], dtype=tensor_type[0][0]),
           tf.TensorSpec(shape=input_shape_tuple[1], dtype=tensor_type[0][1]),))
    exit()
    # create a concrete function to save protobuf graph file
    @tf.function(input_signature=(tf.TensorSpec(shape=input_shape_tuple[0], dtype=tensor_type[0][0]),
                                  tf.TensorSpec(shape=input_shape_tuple[1], dtype=tensor_type[0][1]),))
    def model_sign(input_1, input_2):
        output = model((input_1,input_2))
        final_out = tf.identity(output, name="final_out")
        return final_out

    # def get_flops(model_sign_):
    #     concrete_func = model_sign_.get_concrete_function(tf.TensorSpec(shape=input_shape_tuple[0], dtype=tensor_type[0][0]),
    #                                                  tf.TensorSpec(shape=input_shape_tuple[1], dtype=tensor_type[0][1]))

    #     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    #     with tf.Graph().as_default() as graph:
    #         tf.graph_util.import_graph_def(graph_def, name='')

    #         run_meta = tf.compat.v1.RunMetadata()
    #         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    #         flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

    #         return flops.total_float_ops

    # print(get_flops(model_sign))

    # convert to binary (.pb extension) protobuf
    # with variables converted to constants
    cmsml.tensorflow.save_graph("graph.pb", model_sign, variables_to_constants=True)
    cmsml.tensorflow.save_graph("graph.pb.txt", model_sign, variables_to_constants=True)

    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
        mlflow.log_artifact('graph.pb', 'model_graph')
        mlflow.log_artifact('graph.pb.txt', 'model_graph')
    os.remove('graph.pb')
    os.remove('graph.pb.txt')

if __name__ == '__main__':
    main()