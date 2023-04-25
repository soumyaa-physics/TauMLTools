// #include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Training/interface/DataLoader_tools.h"
#include "TauMLTools/Training/interface/histogram2d.h"

#include "TROOT.h"
#include "TLorentzVector.h"
#include "TMath.h"

// #include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/DisTauTagSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"


struct Data {
    template<typename grid> void init_grid() {
        x[grid::object_type].resize(Setup::n_tau * grid::size * grid::length, 0);
    }
    template<typename T, T... I> Data(std::integer_sequence<T, I...> int_seq)
    : y(Setup::n_tau * Setup::output_classes, 0), tau_i(0), 
    uncompress_index(Setup::n_tau, 0), uncompress_size(0), weights(Setup::n_tau, 0)
    {
        if (Setup::prop_y_glob) y_global.resize(Setup::n_tau * Setup::n_globals, 0);
        ((init_grid<FeaturesHelper<std::tuple_element_t<I, FeatureTuple>>>()),...);
    }
    std::unordered_map<CellObjectType, std::vector<Float_t>> x;
    std::vector<Float_t> y;
    std::vector<Float_t> y_global;
    std::vector<Float_t> weights;

    Long64_t tau_i; // the number of taus filled in the tensor filled_tau <= n_tau;
    std::vector<unsigned long> uncompress_index; // index of the tau when events are dropped;
    Long64_t uncompress_size;
};

// using namespace Setup;

class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
    using JetType = analysis::JetType;
    static constexpr size_t nFeaturesTypes = std::tuple_size_v<FeatureTuple>;

    DataLoader() :
        hasData(false),
        fullData(false),
        hasFile(false)
    { 
        ROOT::EnableThreadSafety();
        if(Setup::n_threads > 1) ROOT::EnableImplicitMT(Setup::n_threads);

        if (Setup::yaxis.size() != (Setup::xaxis_list.size() + 1)){
            throw std::invalid_argument("Y binning list does not match X binning length");
        }

        // auto spectrum_file = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());
        auto file_input = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());
        auto file_target = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());

        Histogram_2D input_histogram ("input" , Setup::yaxis, Setup::xmin, Setup::xmax);
        Histogram_2D target_histogram("target", Setup::yaxis, Setup::xmin, Setup::xmax);

        for (int i = 0; i < Setup::xaxis_list.size(); i++)
        {
            target_histogram.add_x_binning_by_index(i, Setup::xaxis_list[i]);
            input_histogram .add_x_binning_by_index(i, Setup::xaxis_list[i]);
        }

        std::shared_ptr<TH2D> target_th2d = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_target->Get("jet_eta_pt_hist_tau")));
        if (!target_th2d) throw std::runtime_error("Target histogram could not be loaded");

        for( auto const& [tau_type, tau_name] : Setup::tau_types_names)
        {
            std::shared_ptr<TH2D> input_th2d  = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_input ->Get(("jet_eta_pt_hist_"+tau_name).c_str())));
            if (!input_th2d) throw std::runtime_error("Input histogram could not be loaded for jet type "+tau_name);
            target_histogram.th2d_add(*(target_th2d.get()));
            input_histogram .th2d_add(*(input_th2d.get()));

            target_histogram.divide(input_histogram);
            hist_weights[tau_type] = target_histogram.get_weights_th2d(
                ("w_1_"+tau_name).c_str(),
                ("w_1_"+tau_name).c_str()
            );
            if (Setup::debug) hist_weights[tau_type]->SaveAs(("weights_"+tau_name+".root").c_str()); // It's required that all bins are filled in these histograms; save them to check incase binning is too fine and some bins are empty

            target_histogram.reset();
            input_histogram .reset();
        }

        MaxDisbCheck(hist_weights, Setup::weight_thr);
    }

    static void MaxDisbCheck(const std::unordered_map<int ,std::shared_ptr<TH2D>>& hists, Double_t max_thr)
    {
      double min_weight = std::numeric_limits<double>::max();
      double max_weight = std::numeric_limits<double>::lowest();
      for(auto const& [tau_type, hist_] : hists) {
        min_weight = std::min(hist_->GetMinimum(), min_weight);
        max_weight = std::max(hist_->GetMaximum(), max_weight);
      }
      std::cout << "Weights imbalance: " << max_weight / min_weight
                << ", imbalance threshold: " <<  max_thr << std::endl;
      if(max_weight / min_weight > max_thr)
        throw std::runtime_error("The imbalance in the weights exceeds the threshold.");
    }

    const double GetWeight(const int type_id, const double pt, const double eta) const
    {   
        return hist_weights.at(type_id)->GetBinContent(
                    hist_weights.at(type_id)->GetXaxis()->FindFixBin(eta),
                    hist_weights.at(type_id)->GetYaxis()->FindFixBin(pt)
                );
    }


    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    void ReadFile(std::string file_name,
                  Long64_t start_file,
                  Long64_t end_file) // put end_file=-1 to read all events from file
    { 
        tauTuple.reset();
        file = std::make_unique<TFile>(file_name.c_str());
        tauTuple = std::make_unique<tau_tuple::TauTuple>(file.get(), true);
        current_entry = start_file;
        end_entry = tauTuple->GetEntries();
        if(end_file!=-1) end_entry = std::min(end_file, end_entry);
        hasFile = true;
    } 

    bool MoveNext()
    {
        if(!hasFile)
            throw std::runtime_error("File should be loaded with DataLoaderWorker::ReadFile()");

        if(!tauTuple)
            throw std::runtime_error("TauTuple is not loaded!");

        if(!hasData)
        {
            data = std::make_unique<Data>(std::make_index_sequence<nFeaturesTypes>{});
            data->tau_i = 0;
            data->uncompress_size = 0;
            hasData = true;
        }
        while(data->tau_i < Setup::n_tau)
        {
            if(current_entry == end_entry)
            {
                hasFile = false;
                return false;
            }
            
            tauTuple->GetEntry(current_entry);
            auto& tau = const_cast<Tau&>(tauTuple->data());
            const std::optional<JetType> jet_match_type = 
                Setup::recompute_jet_type ? analysis::GetJetType(tau)
                : static_cast<JetType>(tau.tauType); 

            if (jet_match_type)
            {
                if(Setup::prop_y_glob) FillGlobal(tau, data->tau_i, jet_match_type);
                data->y.at(data->tau_i * Setup::output_classes + static_cast<Int_t>(*jet_match_type)) = 1.0;
                data->weights.at(data->tau_i) = GetWeight(static_cast<Int_t>(*jet_match_type), tau.jet_pt, std::abs(tau.jet_eta));
                FillPfCand(tau, data->tau_i);
                data->uncompress_index[data->tau_i] = data->uncompress_size;
                ++(data->tau_i);
            }
            else if ( tau.jet_index >= 0 && Setup::include_mismatched ) {
                if(Setup::prop_y_glob) FillGlobal(tau, data->tau_i, jet_match_type);
                FillPfCand(tau, data->tau_i);
                data->uncompress_index[data->tau_i] = data->uncompress_size;
                ++(data->tau_i);
            }
            ++(data->uncompress_size);
            ++current_entry;
        }
        fullData = true;
        return true;
    }

    bool hasAnyData() {return hasData;}
    
    const Data* LoadData(bool needFull)
    {
      if(!fullData && needFull)
        throw std::runtime_error("Data was not loaded with MoveNext() or array was not fully filled");
        fullData = false;
        hasData = false;
        return data.get();
    }

  private:

    template <typename FeatureT>
    const float Scale(const Int_t idx, const Float_t value)
    {
        return std::clamp((value - FeatureT::mean.at(idx).at(0)) / FeatureT::std.at(idx).at(0),
                            FeatureT::lim_min.at(idx).at(0), FeatureT::lim_max.at(idx).at(0));
    }

    static constexpr Float_t pi = boost::math::constants::pi<Float_t>();

    template <typename Scalar>
    static Scalar DeltaPhi(Scalar phi1, Scalar phi2)
    {
        static constexpr Scalar pi = boost::math::constants::pi<Scalar>();
        Scalar dphi = phi1 - phi2;
        if(dphi > pi)
            dphi -= 2*pi;
        else if(dphi <= -pi)
            dphi += 2*pi;
        return dphi;
    }

    void FillGlobal(const Tau& tau, const Long64_t tau_i,
                    const std::optional<JetType> jet_match_type)
    {
        auto getGlobVecRef = [&](int _fe, Float_t value){
                if(_fe < 0) return;
                const size_t index = Setup::n_globals * tau_i + _fe;
                data->y_global.at(index) = value;
            };

        getGlobVecRef(0, tau.jet_pt);
        getGlobVecRef(1, tau.jet_eta);

        getGlobVecRef(2, -999);
        getGlobVecRef(3, -999);
        getGlobVecRef(4, -999);

        if(jet_match_type)
        {
            if(jet_match_type == JetType::tau)
            {
                reco_tau::gen_truth::GenLepton genLeptons = 
                    reco_tau::gen_truth::GenLepton::fromRootTuple(
                            tau.genLepton_lastMotherIndex,
                            tau.genParticle_pdgId,
                            tau.genParticle_mother,
                            tau.genParticle_charge,
                            tau.genParticle_isFirstCopy,
                            tau.genParticle_isLastCopy,
                            tau.genParticle_pt,
                            tau.genParticle_eta,
                            tau.genParticle_phi,
                            tau.genParticle_mass,
                            tau.genParticle_vtx_x,
                            tau.genParticle_vtx_y,
                            tau.genParticle_vtx_z);

                auto vertex = genLeptons.lastCopy().vertex;
                if( std::abs(genLeptons.lastCopy().pdgId) != 15 )
                    throw std::runtime_error("Error FillGlob: last copy of genLeptons is not tau.");

                // get the displacement wrt to the mother vertex
                auto Lrel = genLeptons.lastCopy().getDisplacement();

                getGlobVecRef(2, Lrel);
                getGlobVecRef(3, std::abs(vertex.rho()));
                getGlobVecRef(4, std::abs(vertex.z()));
            }
        }
    }

    void FillPfCand(const Tau& tau, const Long64_t tau_i)
    {
        
        auto fillGrid = [&](auto _feature, size_t candidate_idx, float value) {
          if(static_cast<int>(_feature) < 0) return; 
          size_t _feature_idx = static_cast<int>(_feature);
          const CellObjectType obj_type = FeaturesHelper<decltype(_feature)>::object_type;
          const size_t start_index =  FeaturesHelper<decltype(_feature)>::length * FeaturesHelper<decltype(_feature)>::size * tau_i;
          const size_t index = start_index + FeaturesHelper<decltype(_feature)>::size * candidate_idx + _feature_idx;
          data->x.at(obj_type).at(index)
                  = Scale<typename  FeaturesHelper<decltype(_feature)>::scaler_type>(_feature_idx, value);
        };        

        const auto getPt = [&](CellObjectType type, size_t index) {
            if(type == CellObjectType::PfCand)
                return tau.pfCand_pt.at(index);
            if(type == CellObjectType::LostTrack)
                return tau.lostTrack_pt.at(index);
            if(type == CellObjectType::SecondaryVertex)
                return tau.sv_pt.at(index);
            throw std::runtime_error("Type of CellObjectType not supported. (getPt)");
        };

        const auto getdR = [&](CellObjectType type, size_t index) {
            if(type == CellObjectType::LostTrack)
                return std::hypot(tau.lostTrack_eta.at(index) - tau.jet_eta,
                                  DeltaPhi(tau.lostTrack_phi.at(index), tau.jet_phi));
            if(type == CellObjectType::SecondaryVertex)
                return std::hypot(tau.sv_eta.at(index) - tau.jet_eta,
                                  DeltaPhi(tau.sv_phi.at(index), tau.jet_phi));
            if(type == CellObjectType::PfCand)
                return std::hypot(tau.pfCand_eta.at(index) - tau.jet_eta,
                              DeltaPhi(tau.pfCand_phi.at(index), tau.jet_phi));
            throw std::runtime_error("Type of CellObjectType not supported. (getdR)");
        };

        const auto get_sequence_idxs = [&](CellObjectType type, size_t n_total, size_t n_max) {
            std::vector<size_t> sequence_idxs(n_total);
            std::iota(sequence_idxs.begin(), sequence_idxs.end(), 0);

            // Drop pfCands that are not mathing within dR < 0.5 to the tau_pt, tau_eta
            sequence_idxs.erase(std::remove_if(sequence_idxs.begin(), sequence_idxs.end(), [&](size_t candidate_idx) {
              return getdR(type, candidate_idx) > Setup::max_dr;
            }), sequence_idxs.end());

            // Sort pfCands by pt
            std::sort(sequence_idxs.begin(), sequence_idxs.end(), [&](size_t a, size_t b) {
              return getPt(type, a) > getPt(type, b);
            });

            // Keep only the first n_max
            if(sequence_idxs.size() > n_max)
                sequence_idxs.resize(n_max);
            
            return sequence_idxs;
        };


        {   // CellObjectType::PfCand

            typedef PfCand_Features Br;

            std::vector<size_t> pfCand_indxs = get_sequence_idxs(CellObjectType::PfCand,
                                                                tau.pfCand_pt.size(),
                                                                Setup::nSeq_PfCand);

            // Iterate over all pfCands and fill the Data
            size_t tensor_i = 0;
            for(size_t pfCand_idx : pfCand_indxs) {

                fillGrid(Br::pfCand_valid,         tensor_i, 1.0);
                fillGrid(Br::pfCand_pt,            tensor_i, tau.pfCand_pt.at(pfCand_idx));
                fillGrid(Br::pfCand_phi,           tensor_i, tau.pfCand_phi.at(pfCand_idx));
                fillGrid(Br::pfCand_log_pt,        tensor_i, std::log(tau.pfCand_pt.at(pfCand_idx)));
                fillGrid(Br::pfCand_log_E,         tensor_i,
                    std::log(std::sqrt(std::pow(tau.pfCand_mass.at(pfCand_idx),2) + 
                                       std::pow(tau.pfCand_pt.at(pfCand_idx),2) *
                                       std::pow(std::cosh(tau.pfCand_eta.at(pfCand_idx)),2)
                                       ))
                    );

                fillGrid(Br::pfCand_charge,           tensor_i, tau.pfCand_charge.at(pfCand_idx));
                fillGrid(Br::pfCand_puppiWeight,      tensor_i, tau.pfCand_puppiWeight.at(pfCand_idx));
                fillGrid(Br::pfCand_puppiWeightNoLep, tensor_i, tau.pfCand_puppiWeightNoLep.at(pfCand_idx));
                fillGrid(Br::pfCand_lostInnerHits,    tensor_i, tau.pfCand_lostInnerHits.at(pfCand_idx));
                fillGrid(Br::pfCand_nPixelHits,       tensor_i, tau.pfCand_nPixelHits.at(pfCand_idx));
                fillGrid(Br::pfCand_nHits,            tensor_i, tau.pfCand_nHits.at(pfCand_idx));
                
                if( tau.pfCand_hasTrackDetails.at(pfCand_idx) )
                {   
                    fillGrid(Br::pfCand_hasTrackDetails, tensor_i, tau.pfCand_hasTrackDetails.at(pfCand_idx));
                    
                    fillGrid(Br::pfCand_dz,      tensor_i, tau.pfCand_dz.at(pfCand_idx));
                    fillGrid(Br::pfCand_dz_sig,  tensor_i, tau.pfCand_dz.at(pfCand_idx)/tau.pfCand_dz_error.at(pfCand_idx));
                    fillGrid(Br::pfCand_dxy,     tensor_i, tau.pfCand_dxy.at(pfCand_idx));
                    fillGrid(Br::pfCand_dxy_sig, tensor_i, tau.pfCand_dxy.at(pfCand_idx)/tau.pfCand_dxy_error.at(pfCand_idx));

                    if( tau.pfCand_track_ndof.at(pfCand_idx) > 0 )
                    {
                        fillGrid(Br::pfCand_chi2_ndof, tensor_i, tau.pfCand_track_chi2.at(pfCand_idx)/tau.pfCand_track_ndof.at(pfCand_idx));
                        fillGrid(Br::pfCand_ndof, tensor_i, tau.pfCand_track_ndof.at(pfCand_idx));
                    }
              
                }
                
                fillGrid(Br::pfCand_caloFraction,         tensor_i, tau.pfCand_caloFraction.at(pfCand_idx));
                fillGrid(Br::pfCand_hcalFraction,         tensor_i, tau.pfCand_hcalFraction.at(pfCand_idx));
                fillGrid(Br::pfCand_rawCaloFraction,      tensor_i, tau.pfCand_rawCaloFraction.at(pfCand_idx));
                fillGrid(Br::pfCand_rawHcalFraction,      tensor_i, tau.pfCand_rawHcalFraction.at(pfCand_idx));
                
                fillGrid(Br::pfCand_particleType,         tensor_i, tau.pfCand_particleType.at(pfCand_idx));
                fillGrid(Br::pfCand_pvAssociationQuality, tensor_i, tau.pfCand_pvAssociationQuality.at(pfCand_idx));
                fillGrid(Br::pfCand_fromPV,               tensor_i, tau.pfCand_fromPV.at(pfCand_idx));

                fillGrid(Br::pfCand_deta, tensor_i, tau.pfCand_eta.at(pfCand_idx) - tau.jet_eta);
                fillGrid(Br::pfCand_dphi, tensor_i, DeltaPhi<Float_t>(tau.pfCand_phi.at(pfCand_idx), tau.jet_phi));

                ++tensor_i;
            }
        }

        {   // CellObjectType::LostTrack

            typedef LostTrack_Features Br;

            std::vector<size_t> lostTrack_indxs = get_sequence_idxs(CellObjectType::LostTrack,
                                                                tau.lostTrack_pt.size(),
                                                                Setup::nSeq_LostTrack);

            // Iterate over all lostTracks and fill the Data
            size_t tensor_i = 0;
            for(size_t lostTrack_idx : lostTrack_indxs) {

                fillGrid(Br::lostTrack_valid,         tensor_i, 1.0);
                fillGrid(Br::lostTrack_pt,            tensor_i, tau.lostTrack_pt.at(lostTrack_idx));
                fillGrid(Br::lostTrack_phi,           tensor_i, tau.lostTrack_phi.at(lostTrack_idx));
                fillGrid(Br::lostTrack_log_pt,        tensor_i, std::log(tau.lostTrack_pt.at(lostTrack_idx)));
                fillGrid(Br::lostTrack_log_E,         tensor_i,
                    std::log(std::sqrt(pow(tau.lostTrack_mass.at(lostTrack_idx),2) +
                                       pow(tau.lostTrack_pt.at(lostTrack_idx),2) *
                                       pow(std::cosh(tau.lostTrack_eta.at(lostTrack_idx)),2)))
                    );

                fillGrid(Br::lostTrack_charge,        tensor_i, tau.lostTrack_charge.at(lostTrack_idx));
                fillGrid(Br::lostTrack_nPixelHits,    tensor_i, tau.lostTrack_nPixelHits.at(lostTrack_idx));
                fillGrid(Br::lostTrack_nHits,         tensor_i, tau.lostTrack_nHits.at(lostTrack_idx));

                if( tau.lostTrack_hasTrackDetails.at(lostTrack_idx) )
                {   
                    fillGrid(Br::lostTrack_hasTrackDetails, tensor_i, tau.lostTrack_hasTrackDetails.at(lostTrack_idx));
                    
                    fillGrid(Br::lostTrack_dz,        tensor_i, tau.lostTrack_dz.at(lostTrack_idx));
                    fillGrid(Br::lostTrack_dz_sig,    tensor_i, tau.lostTrack_dz.at(lostTrack_idx)/tau.lostTrack_dz_error.at(lostTrack_idx));
                    fillGrid(Br::lostTrack_dxy,       tensor_i, tau.lostTrack_dxy.at(lostTrack_idx));
                    fillGrid(Br::lostTrack_dxy_sig,   tensor_i, tau.lostTrack_dxy.at(lostTrack_idx)/tau.lostTrack_dxy_error.at(lostTrack_idx));

                    if( tau.lostTrack_track_ndof.at(lostTrack_idx) > 0 )
                    {
                        fillGrid(Br::lostTrack_chi2_ndof, tensor_i, tau.lostTrack_track_chi2.at(lostTrack_idx)/tau.lostTrack_track_ndof.at(lostTrack_idx));
                        fillGrid(Br::lostTrack_ndof, tensor_i, tau.lostTrack_track_ndof.at(lostTrack_idx));
                    }
              
                }

                fillGrid(Br::lostTrack_particleType,         tensor_i, tau.lostTrack_particleType.at(lostTrack_idx));
                fillGrid(Br::lostTrack_pvAssociationQuality, tensor_i, tau.lostTrack_pvAssociationQuality.at(lostTrack_idx));
                fillGrid(Br::lostTrack_fromPV,               tensor_i, tau.lostTrack_fromPV.at(lostTrack_idx));

                fillGrid(Br::lostTrack_deta, tensor_i, tau.lostTrack_eta.at(lostTrack_idx) - tau.jet_eta);
                fillGrid(Br::lostTrack_dphi, tensor_i, DeltaPhi<Float_t>(tau.lostTrack_phi.at(lostTrack_idx), tau.jet_phi));

                ++tensor_i;
            }
        }

        {   //CellObjectType::SecondaryVertex
        
            typedef SecondaryVertex_Features Br;

            std::vector<size_t> sv_indxs = get_sequence_idxs(CellObjectType::SecondaryVertex,
                                                                tau.sv_pt.size(),
                                                                Setup::nSeq_SecondaryVertex);

            // Iterate over all SVs and fill the Data
            size_t tensor_i = 0;
            for(size_t sv_idx : sv_indxs) {

                fillGrid(Br::sv_valid,         tensor_i, 1.0);
                fillGrid(Br::sv_pt,            tensor_i, tau.sv_pt.at(sv_idx));
                fillGrid(Br::sv_phi,           tensor_i, tau.sv_phi.at(sv_idx));
                fillGrid(Br::sv_log_pt,        tensor_i, std::log(tau.sv_pt.at(sv_idx)));
                fillGrid(Br::sv_log_E,         tensor_i,
                    std::log(std::sqrt(pow(tau.sv_mass.at(sv_idx),2) +
                                       pow(tau.sv_pt.at(sv_idx),2) *
                                       pow(std::cosh(tau.sv_eta.at(sv_idx)),2)))
                    );

                fillGrid(Br::sv_dx,             tensor_i, tau.sv_x.at(sv_idx) - tau.pv_x);
                fillGrid(Br::sv_dy,             tensor_i, tau.sv_y.at(sv_idx) - tau.pv_y);
                fillGrid(Br::sv_dz,             tensor_i, tau.sv_z.at(sv_idx) - tau.pv_z);
                fillGrid(Br::sv_chi2_ndof,      tensor_i, tau.sv_chi2.at(sv_idx)/tau.sv_ndof.at(sv_idx));
                fillGrid(Br::sv_ndof,           tensor_i, tau.sv_ndof.at(sv_idx));

                fillGrid(Br::sv_deta, tensor_i, tau.sv_eta.at(sv_idx) - tau.jet_eta);
                fillGrid(Br::sv_dphi, tensor_i, DeltaPhi<Float_t>(tau.sv_phi.at(sv_idx), tau.jet_phi));

                ++tensor_i;
            }

        }
    }

private:

  Long64_t end_entry;
  Long64_t current_entry; // number of the current entry in the file
  Long64_t current_tau; // number of the current tau candidate

  bool hasData;
  bool fullData;
  bool hasFile;

  std::unique_ptr<TFile> file; // to open with one file
  std::unique_ptr<TauTuple> tauTuple;
  std::unique_ptr<Data> data;
  std::unordered_map<int ,std::shared_ptr<TH2D>> hist_weights;

};
