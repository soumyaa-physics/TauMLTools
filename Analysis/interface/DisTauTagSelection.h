#pragma once

#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Core/interface/exception.h"
#include "TauMLTools/Core/interface/EnumNameMap.h"

namespace JetTypeSelection {

static const double jet_pt = 20; //GeV
static const double jet_eta = 2.4;

static const double genJet_pt = 20; //GeV
static const double genJet_eta = 2.4;

static const double gen_tau_pt = 20; //GeV
static const double gen_tau_eta = 2.4; //GeV
static const double gen_z = 100; //cm
static const double gen_rho = 50; //cm
static const double gen_rho_min = -999; //cm
static const double genLepton_jet_dR = 0.4;

}

namespace analysis {

enum class JetType { jet = 0, tau = 1 };

ENUM_NAMES(JetType) = {
    { JetType::jet, "jet" },
    { JetType::tau, "tau" }
};

std::optional<JetType> GetJetType(const tau_tuple::Tau& tau,
                                  const bool enable_pu_jets = false,
                                  const bool take_all_taus = false) {

    using GTGenLeptonKind = reco_tau::gen_truth::GenLepton::Kind;

    if( tau.jet_index >= 0 && 
        tau.jet_pt > JetTypeSelection::jet_pt &&
        std::abs(tau.jet_eta) < JetTypeSelection::jet_eta  ) 
    {

        if( tau.genLepton_kind == static_cast<int>(GTGenLeptonKind::TauDecayedToHadrons) )
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
            auto visible_p4 = genLeptons.visibleP4();

            if( std::abs(genLeptons.lastCopy().pdgId) != 15 )
                throw exception("Error GetJetType: last copy of genLeptons is not tau.");

            if( std::abs(vertex.z()) < JetTypeSelection::gen_z &&
                std::abs(vertex.rho()) < JetTypeSelection::gen_rho &&
                std::abs(vertex.rho()) > JetTypeSelection::gen_rho_min &&
                visible_p4.pt() > JetTypeSelection::gen_tau_pt &&
                std::abs(visible_p4.eta()) < JetTypeSelection::gen_tau_eta 
            )
            {
                ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> jet_p4
                    ( tau.jet_pt, tau.jet_eta, tau.jet_phi, tau.jet_mass );

                double dR = ROOT::Math::VectorUtil::DeltaR(jet_p4, genLeptons.visibleP4());
                if( dR < JetTypeSelection::genLepton_jet_dR ) return JetType::tau;
            } else if( take_all_taus &&
                       visible_p4.pt() > JetTypeSelection::gen_tau_pt &&
                       std::abs(visible_p4.eta()) < JetTypeSelection::gen_tau_eta ) {
                       return JetType::tau;
                       }

        } else if( tau.genJet_index >= 0 && 
                   (tau.genLepton_kind <= 0 || tau.genLepton_kind == static_cast<int>(GenLeptonMatch::NoMatch)) &&
                   tau.genJet_pt > JetTypeSelection::genJet_pt && 
                   std::abs(tau.genJet_eta) < JetTypeSelection::genJet_eta ) {
            return JetType::jet;
        } else if ( enable_pu_jets && tau.genJet_index < 0 ) {
            return JetType::jet;
        }
    }
    return std::nullopt;

}
} // namespace analysis