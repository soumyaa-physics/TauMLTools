#!/bin/bash

set -e

python plot_wp_eff.py vs_type=jet from_skims=False output_skim_folder=wp_eff/data/TT_STAU250GeV wp_eff_var@_global_=jet_pt 'create_df.path_to_mlflow=../Training/python/distautag/mlruns/' create_df.experiment_id=7 create_df.run_id=a27159734e304ea4b7f9e0042baa9e22
python plot_wp_eff.py vs_type=jet from_skims=False output_skim_folder=wp_eff/data/TT_STAU250GeV wp_eff_var@_global_=jet_eta 'create_df.path_to_mlflow=../Training/python/distautag/mlruns/' create_df.experiment_id=7 create_df.run_id=a27159734e304ea4b7f9e0042baa9e22
python plot_wp_eff.py vs_type=jet from_skims=False output_skim_folder=wp_eff/data/TT_STAU250GeV wp_eff_var@_global_=Lrel 'create_df.path_to_mlflow=../Training/python/distautag/mlruns/' create_df.experiment_id=7 create_df.run_id=a27159734e304ea4b7f9e0042baa9e22