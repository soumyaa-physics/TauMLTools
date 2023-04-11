# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import time
import config_parse
import os
import yaml
R.gROOT.SetBatch(True)

PLOT_HISTOGRAM_OUTPUT = True

R.gROOT.ProcessLine(".include ../../..")

print("Compiling Setup classes...")

with open(os.path.abspath( "../configs/trainingDisTauTag_v2.yaml")) as f:
    config = yaml.safe_load(f)

R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_vDisTauTag_v2.json", config, verbose=False))
R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "TauMLTools/Analysis/interface/TauTuple.h"')
R.gInterpreter.Declare('#include "../interface/DataLoaderDisTauTag_v2_main.h"')


n_tau          = config["Setup"]["n_tau"]
outclass       = len(config["Setup"]["jet_types_names"])
n_features = {}
n_features_map = {}
n_seq = {}
for celltype in config["Features_all"]:
    n_features[str(celltype)] = len(config["Features_all"][celltype]) - \
                                len(config["Features_disable"][celltype])
    if celltype in config["SequenceLength"]: # sequence length
        n_seq[str(celltype)] = config["SequenceLength"][celltype]
    
    n_features_map[celltype] = {}
    enume = 0
    for feature in config["Features_all"][celltype]:
        feature_str = next(iter(feature))
        if feature_str not in config["Features_disable"][celltype]:
            n_features_map[celltype][feature_str] = enume
            enume += 1
            
input_grids = config["CellObjectType"]

input_files = []
for root, dirs, files in os.walk(os.path.abspath(R.Setup.input_dir)):
    for file in files:
        input_files.append(os.path.join(root, file))

data_loader = R.DataLoader()

n_batches = 50

times = []

file_i = 0
data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
file_i += 1

def getdata(_obj_f, tau_i, _reshape, _dtype=np.float32):
    x = np.copy(np.frombuffer(_obj_f.data(), dtype=_dtype, count=_obj_f.size()))
    return x[:tau_i] if _reshape==-1 else x.reshape(_reshape)[:tau_i]

def getsequence(_obj_grid,
                _tau_i,
                _batch_size,
                _input_grids,
                _n_features,
                _n_seq):
    return [ getdata(_obj_grid[getattr(R.CellObjectType,group)], _tau_i,
            (_batch_size, _n_seq[group], _n_features[group]))
            for group in _input_grids]

def create_histograms(conf):
    
    hists = {}
    '''Create TH1 histograms from the input config'''
    for domain in [0, 1]: #signal/background
        hists[domain] = {}
        for celltype in conf["Features_all"]:
            hists[domain][celltype] = {}
            for feature in conf["Features_all"][celltype]:
                feature_str = list(feature)[0]
                if feature_str in conf["Features_disable"][celltype]:
                    continue
                hists[domain][celltype][feature_str] = R.TH1D(celltype+feature_str+"_data"+str(domain),
                                                              celltype+feature_str+"_data"+str(domain),
                                                                        50, -2, 2)
    return hists


def test_input():
    hists_ditionary = create_histograms(config)
    for i in range(n_batches):
        # if i % 100 == 0:
        print(i)
        start = time.time()
        checker = data_loader.MoveNext()
        if checker==False:
            data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
            file_i += 1
            continue
        
        data = data_loader.LoadData(checker)
        
        X = getsequence(data.x, data.tau_i, n_tau, input_grids, n_features, n_seq)
        Y = getdata(data.y, data.tau_i, (n_tau, outclass))
        W = getdata(data.weights, data.tau_i,  -1)

    
        for x_i, input_ in enumerate(X):
            for tau_i in  range(n_tau):
                celltype = input_grids[x_i]
                for feature in n_features_map[celltype]:
                    domian_i = int(Y[tau_i][1] == 1)
                    # print("Y[tau_i]: ", Y[tau_i])
                    # print("domian_i: ", domian_i)
                    # print("celltype: ", celltype)
                    # print("feature: ", feature, n_features_map[celltype][feature])
                    # print("sequence length: ", n_seq[celltype])
                    for pf_i in range(n_seq[celltype]):
                        # print("pf_i: ", pf_i)
                        if input_[tau_i][pf_i][0] == 0:
                            continue
                        hists_ditionary[domian_i][celltype][feature].Fill(input_[tau_i][pf_i][n_features_map[celltype][feature]])
        # exit()
    # check if histograms foler exists if not create it
    if not os.path.exists("histograms"):
        os.makedirs("histograms")

    # plot histograms on canvas and save with the naming coresponing to the feature
    for celltype in hists_ditionary[0]:
        for feature in hists_ditionary[0][celltype]:
            
            c = R.TCanvas("c", "c", 800, 800)
            
            hists_ditionary[0][celltype][feature].Scale(1/hists_ditionary[0][celltype][feature].Integral())
            hists_ditionary[0][celltype][feature].SetLineColor(R.kBlue)
            hists_ditionary[0][celltype][feature].SetFillColor(R.kBlue)
            hists_ditionary[0][celltype][feature].SetLineWidth(1)
            hists_ditionary[0][celltype][feature].SetMarkerSize(0)
            hists_ditionary[0][celltype][feature].SetMaximum(1.0)
            hists_ditionary[0][celltype][feature].SetMinimum(0.0)
            hists_ditionary[0][celltype][feature].SetFillStyle(3353)
            hists_ditionary[0][celltype][feature].Draw("HIST")

            hists_ditionary[1][celltype][feature].Scale(1/hists_ditionary[1][celltype][feature].Integral())
            hists_ditionary[1][celltype][feature].SetLineColor(R.kRed)
            hists_ditionary[1][celltype][feature].SetFillColor(R.kRed)
            hists_ditionary[1][celltype][feature].SetLineWidth(1)
            hists_ditionary[1][celltype][feature].SetMarkerSize(0)
            hists_ditionary[1][celltype][feature].SetMaximum(1.0)
            hists_ditionary[1][celltype][feature].SetMinimum(0.0)
            hists_ditionary[1][celltype][feature].SetFillStyle(3335)
            hists_ditionary[1][celltype][feature].Draw("HIST SAME")

            legend = R.TLegend(0.73,0.73,1.0,1.0)
            legend.AddEntry(hists_ditionary[0][celltype][feature],"Background","l")
            legend.AddEntry(hists_ditionary[1][celltype][feature],"Signal","l")
            legend.Draw("same")

            c.SaveAs("histograms/"+ celltype + "_" + feature + ".png")

# Create histogram for every variable:
if PLOT_HISTOGRAM_OUTPUT:
    print("Creating histograms with features...")
    test_input()
    exit()


for i in range(n_batches):

    start = time.time()
    checker = data_loader.MoveNext()
    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
       file_i += 1
       continue
    
    data = data_loader.LoadData(checker)
    
    X = getsequence(data.x, data.tau_i, n_tau, input_grids, n_features, n_seq)
    # X_glob = getdata(data.x_glob, data.tau_i, (n_tau, n_features["Global"]))
    Y = getdata(data.y, data.tau_i, (n_tau, outclass))
    W = getdata(data.weights, data.tau_i,  -1)

    for x_i, input_ in  enumerate(X):
        hists[x_i]
        
    print("Y:\n",Y[0])
    print("W:\n",W[0])


    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))
