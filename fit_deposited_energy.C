// ROOT macro to analyze deposited energy from MCParticle objects
// Usage:
//   root -l -b -q '/home/claire/allpix-squared/SiC_3x3/fit_deposited_energy.C("...root")'
// Optional second argument is accepted for drop-in compatibility
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1D.h>
#include <TF1.h>
#include <TSystem.h>
#include <TROOT.h>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <limits>

// loads the allpix library
R__LOAD_LIBRARY(/home/claire/allpix-squared/lib/libAllpixObjects.so)
#include "/home/claire/allpix-squared/src/objects/MCParticle.hpp"

// filename is the ROOT file
void fit_deposited_energy(const char* filename, double /*unused*/ = 0.0) {

    // makes ROOT run the shared allpix library
    gSystem->Load("/home/claire/allpix-squared/lib/libAllpixObjects.so");

    printf("MACRO_SCRIPT---->IMPORTED ROOT FILE IS %s\n", filename);

    TFile f(filename, "READ"); // Opens the ROOT file

    // check if file failed to open
    if(f.IsZombie()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Retrieve the MCParticle tree
    TTree* t = static_cast<TTree*>(f.Get("MCParticle"));
    if(t == nullptr) {
        std::cerr << "Tree 'MCParticle' not found in file." << std::endl;
        return;
    }

    // determines the correct branch name
    std::string detector_branch = "thedetector";

    // if branch is unavailable use first available branch instead
    if(t->GetBranch(detector_branch.c_str()) == nullptr) {
        if(t->GetListOfBranches() != nullptr && t->GetListOfBranches()->GetEntries() > 0) {
            detector_branch = t->GetListOfBranches()->At(0)->GetName();
        } else {
            std::cerr << "No detector branches found in tree 'MCParticle'." << std::endl;
            return;
        }
    }

    // prints the branch used
    std::cout << "Using MCParticle branch: " << detector_branch << std::endl;

    // creates a reader for the tree
    TTreeReader reader("MCParticle", &f);
    TTreeReaderValue<std::vector<allpix::MCParticle*>> particles(reader, detector_branch.c_str());

    // Initialize statistics
    double min_e = std::numeric_limits<double>::infinity();
    double max_e = 0.0;
    unsigned long long total_events = 0;  // all events
    unsigned long long depositing_events = 0;  // events with nonzero deposited energy

    // determines range of histogram
    while(reader.Next()) {
        total_events++;
        double event_e_mev = 0.0;
        for(auto* p : *particles) {
            if(p == nullptr) {
                continue;
            }
            event_e_mev += p->getTotalDepositedEnergy();
        }
        if(event_e_mev <= 0.0) {
            continue;
        }
        depositing_events++;
        const double e_keV = event_e_mev * 1.0e3;
        min_e = std::min(min_e, e_keV);
        max_e = std::max(max_e, e_keV);
    }

    const bool has_signal = std::isfinite(min_e) && max_e > 0.0;
    if(!has_signal) {
        std::cerr << "No deposited energy entries found." << std::endl;
    }

    double low_e = has_signal ? std::max(0.0, min_e * 0.95) : 0.0;
    double high_e = has_signal ? max_e * 1.05 : 1.0;

    // define histogram ranges
    if(!(high_e > low_e)) {
        low_e = std::max(0.0, min_e - 10.0);
        high_e = max_e + 10.0;
    }

    // builds output file names
    // builds _energy_hist.csv, _stats.csv, _histograms.root
    const std::string input_name(filename);
    const auto dot_pos = input_name.rfind(".root");
    const std::string base_name = (dot_pos == std::string::npos) ? input_name : input_name.substr(0, dot_pos);
    const std::string energy_csv = base_name + "_energy_hist.csv";
    const std::string stats_csv = base_name + "_stats.csv";
    const std::string out_hist_root = base_name + "_histograms.root";

    // creates energy histogram
    TH1D h_energy("h_deposited_energy", "Total deposited energy per event;energy [keV];entries", 200, low_e, high_e);

    // creates another reader and repeats the loop to fill the histogram
    TTreeReader reader2("MCParticle", &f);
    TTreeReaderValue<std::vector<allpix::MCParticle*>> particles2(reader2, detector_branch.c_str());
    while(reader2.Next()) {
        double event_e_mev = 0.0;
        for(auto* p : *particles2) {
            if(p == nullptr) {
                continue;
            }
            event_e_mev += p->getTotalDepositedEnergy();
        }
        if(event_e_mev <= 0.0) {
            continue;
        }
        const double e_keV = event_e_mev * 1.0e3;
        h_energy.Fill(e_keV);  // fills the energy histogram
    }

    // Fits the histogram with a gaussian curve
    // if(h_energy.GetEntries() > 0) {
    //     h_energy.Fit("gaus");
    // }

    {
        // export energy histogram to csv
        std::ofstream out(energy_csv);
        out << "bin_center,bin_low,bin_high,count\n";
        for(int i = 1; i <= h_energy.GetNbinsX(); ++i) {
            out << h_energy.GetXaxis()->GetBinCenter(i) << ","
                << h_energy.GetXaxis()->GetBinLowEdge(i) << ","
                << h_energy.GetXaxis()->GetBinUpEdge(i) << ","
                << h_energy.GetBinContent(i) << "\n";
        }
    }
    {
        // exports statistics (keep column names compatible with existing scripts)
        std::ofstream out(stats_csv);
        out << "total_particles,depositing_particles,efficiency\n";
        const double eff = total_events > 0 ? static_cast<double>(depositing_events) / total_events : 0.0;
        out << total_events << "," << depositing_events << "," << eff << "\n";
        printf("MACRO_SCRIPT------>STATS_CSV containing event info is %s\n", stats_csv.c_str());
    }
    {
        // saves histogram to ROOT file
        TFile out_root(out_hist_root.c_str(), "RECREATE");
        h_energy.Write();
        out_root.Close();
        printf("MACRO_SCRIPT---->ROOT file where histogram is saved is %s\n", out_hist_root.c_str());
    }

    // prints gaussian fit results
    // if(auto* fit_e = h_energy.GetFunction("gaus")) {
    //     std::cout << "Deposited-energy Gaussian mean = " << fit_e->GetParameter(1)
    //               << " keV, sigma = " << fit_e->GetParameter(2) << " keV" << std::endl;
    // }
}
