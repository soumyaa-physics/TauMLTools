#!/bin/bash

set -e

# Script setups:

allID=(
    89d74e3d6851467c8c7b5d814384b90f
    # a27159734e304ea4b7f9e0042baa9e22
    # 77a1148fffdf4a7e8b9699b2202f7a35
    # ee5dfec0e03a4dd2a27149b5a611eb7c
    # a6eabb83ccb345d3b4f07e13a1c72a2d
    # 145c6108c09640f4b61670bd4d40fded
)

# assigned for evaluation:
STAU_FILES="input_filename=[eventTuple_44,eventTuple_90,eventTuple_45,eventTuple_46,eventTuple_92,eventTuple_47,eventTuple_93,eventTuple_94,eventTuple_49]"

# used in training:
# STAU_FILES="input_filename=[eventTuple_109, eventTuple_117, eventTuple_77, eventTuple_60, eventTuple_100, eventTuple_57, eventTuple_69, eventTuple_76]"

# TTBAR_FILES="input_filename=[eventTuple_10,eventTuple_19,eventTuple_27,eventTuple_35]"
TTBAR_FILES="input_filename=[eventTuple_10,eventTuple_19]"

# PU_TTBAR_FILES=(
#     eventTuple_1357
#     eventTuple_1482
#     eventTuple_1607
#     eventTuple_1732
#     eventTuple_1857
#     eventTuple_1982
# )

EXP_ID=7

# DATASET_MIX_ALIAS=NEW-RUN-TTSemiLept-STAU400_100mm
DATASET_MIX_ALIAS=NEW-RUN-TTSemiLept-HNL
# DATASET_MIX_ALIAS=ISR-STAU400_100mm
# DATASET_MIX_ALIAS=PU-TTBar-STAU400_100mm

#Apply training to all models
for ((id_i=1;id_i<=${#allID[@]};id_i++))
do
    # python apply_training.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} path_to_input_dir=/pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau400_lsp1_ctau100mm_v6/crab_STAU_longlived_M400_10cm_v6/230226_140725/0000/ $STAU_FILES sample_alias=STAU_M400_100mm-new-ntuples_train_sample

    python apply_training.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} path_to_input_dir=/pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/crab_TTToSemiLeptonic/230221_221903/0000/ $TTBAR_FILES sample_alias=TTToSemiLeptonic-new-ntuples

    # python apply_training.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} path_to_input_dir=/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/DisTauTag_prod2023/TauMLTools/HNL-samples/HeavyNeutrino_trilepton_M-10_V-0.0004_tau_massiveAndCKM_LO/ 'input_filename=HeavyNeutrino_trilepton_M-10_V-0.0004_tau_massiveAndCKM_LO' sample_alias=HeavyNeutrino_trilepton_M-10_V-0.0004_tau_massiveAndCKM_LO

    # python apply_training.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} path_to_input_dir=/pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/crab_TTToSemiLeptonic/230221_221903/0000/ $TTBAR_FILES sample_alias=TTToSemiLeptonic-new-ntuples-plus-PU 'training_cfg_upd.Setup.get_pu_jets=True'

done

# Evaluate all models
# for ((id_i=0;id_i<${#allID[@]};id_i++))
# do
#     echo "Evaluation for" ${allID[$id_i]}
#     python evaluate_performance.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} discriminator=DisTauTag_v2 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS}

# done

# Plot all bins
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.0, 200.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.0, 0.2]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.2, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[5.0, 10.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[10.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[50.0, 200.0]' 

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.0, 200.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.0, 0.2]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[0.2, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[5.0, 10.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[10.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=1 vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lxy=[50.0, 200.0]'

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 0.2]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.2, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 10.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[10.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet dataset_alias=${DATASET_MIX_ALIAS} 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'


# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'
