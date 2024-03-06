#!/bin/bash
set -e

# Following script run evaluation machinary for all mass points for a final

allID=(
    # 89d74e3d6851467c8c7b5d814384b90f
    # a27159734e304ea4b7f9e0042baa9e22
    # 145c6108c09640f4b61670bd4d40fded
    a6eabb83ccb345d3b4f07e13a1c72a2d
)

paths=(
    # /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau1000mm_v4/crab_STAU_longlived_M100/230226_164516/0000/
    # /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau250_lsp1_ctau1000mm_v4/crab_STAU_longlived_M250/230226_140706/0000/
    # /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau400_lsp1_ctau1000mm_v4/crab_STAU_longlived_M400/230226_140711/0000/
    # /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6/crab_STAU_longlived_M100_10cm_v6/230226_140716/0000/
    /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau250_lsp1_ctau100mm_v6/crab_STAU_longlived_M250_10cm_v6/230226_140720/0000/
    # /pnfs/desy.de/cms/tier2/store/user/myshched/new-ntuples-tau-pog-v4_ext1/SUS-RunIISummer20UL18GEN-stau400_lsp1_ctau100mm_v6/crab_STAU_longlived_M400_10cm_v6/230226_140725/0000/
)

names=(
    # stau100_lsp1_ctau1000mm
    # stau250_lsp1_ctau1000mm
    # stau400_lsp1_ctau1000mm
    # stau100_lsp1_ctau100mm
    stau250_lsp1_ctau100mm
    # stau400_lsp1_ctau100mm
)

files=(
    eventTuple_41
    eventTuple_47
    # eventTuple_51
    # eventTuple_62
    # eventTuple_68
    # eventTuple_72
    # eventTuple_78
    # eventTuple_79
    # eventTuple_89
)
# files_char="input_filename=[eventTuple_41,eventTuple_47,eventTuple_51,eventTuple_62,eventTuple_68,eventTuple_72,eventTuple_78,eventTuple_79,eventTuple_89]"
files_char="input_filename=[eventTuple_41,eventTuple_47]"

EXP_ID=7

### Check the existence of the files
# for ((path_i=0;path_i<${#paths[@]};path_i++))
# do
#     for ((file_i=0;file_i<${#files[@]};file_i++))
#     do
#         file="${paths[$path_i]}/${files[$file_i]}.root";
#         if ! test -f ${file}; then
#             echo "File does not exist." ${file}
#             exit 0
#         fi
#     done
# done

# Apply training to all models
 for ((id_i=0;id_i<${#allID[@]};id_i++))
 do
     # Apply training to all datasets
     for ((path_i=0;path_i<${#paths[@]};path_i++))
     do
         python apply_training.py \
             path_to_mlflow=../Training/python/distautag/mlruns \
             experiment_id=${EXP_ID} \
             run_id=${allID[$id_i]} \
             path_to_input_dir=${paths[$path_i]} \
             $files_char \
             sample_alias=${names[$path_i]}
     done
 done

# Evaluate training to all models
 for ((id_i=0;id_i<${#allID[@]};id_i++))
 do
     # Apply training to all datasets
     for ((path_i=0;path_i<${#paths[@]};path_i++))
     do
         mix="input_samples=\"{TTToSemiLeptonic-new-ntuples: ['jet'], ${names[$path_i]}: ['tau']}\""
         mix_name="MIX_TTSemiLep_${names[$path_i]}"
         echo $mix
         echo $mix_name
         eval "python evaluate_performance.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} run_id=${allID[$id_i]} discriminator=DisTauTag_v2 vs_type=jet dataset_alias=${mix_name} $mix"
     done
 done

# python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]' 

python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 50]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[50, 100]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[100, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'

python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 200.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[0.0, 1.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[1.0, 5.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[5.0, 20.0]' 
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[20.0, 50.0]'
python plot_roc.py path_to_mlflow=../Training/python/distautag/mlruns experiment_id=${EXP_ID} vs_type=jet 'bin.jet_pt=[20, 1000]' 'bin.jet_eta=[-2.3, 2.3]' 'bin.Lrel=[50.0, 200.0]'
