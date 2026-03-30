// ROOT macro to analyze collected charge/energy from PixelCharge objects
// Usage:
//   root -l -b -q '/home/claire/allpix-squared/SiC_3x3/fit_collected_charge.C("...root",3.9)'
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1D.h>
#include <TF1.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TSystem.h>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

//loads the allpix library
R__LOAD_LIBRARY(/home/claire/allpix-squared/lib/libAllpixObjects.so)
#include "/home/claire/allpix-squared/src/objects/PixelCharge.hpp"

//filename is the ROOT file
void fit_collected_charge(const char* filename, double ehpair_eV) {

    //makes ROOT run the shared allpix library
    gSystem->Load("/home/claire/allpix-squared/lib/libAllpixObjects.so");

    printf("MACRO_SCRIPT---->IMPORTED ROOT FILE IS %s\n", filename);

    TFile f(filename, "READ"); //Opens the ROOT file

    //check if file failed to open
    if(f.IsZombie()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    //Retrieve the pixel charge tree
    TTree* t = static_cast<TTree*>(f.Get("PixelCharge"));
    if(t == nullptr) {
        std::cerr << "Tree 'PixelCharge' not found in file." << std::endl;
        return;
    }

    //determines the correct branch name
    std::string detector_branch = "thedetector";

    //if branch is unavailable use first available branch instead
    if(t->GetBranch(detector_branch.c_str()) == nullptr) {
        if(t->GetListOfBranches() != nullptr && t->GetListOfBranches()->GetEntries() > 0) {
            detector_branch = t->GetListOfBranches()->At(0)->GetName();
        } else {
            std::cerr << "No detector branches found in tree 'PixelCharge'." << std::endl;
            return;
        }
    }

    //prints the branch used
    std::cout << "Using PixelCharge branch: " << detector_branch << std::endl;

    //creates a reader for the tree
    TTreeReader reader("PixelCharge", &f);
    //gets the number of pixel charges in each event
    TTreeReaderValue<std::vector<allpix::PixelCharge*>> pixels(reader, detector_branch.c_str());


    //Initialize statistics
    double min_q = std::numeric_limits<double>::infinity(); //min collected charge per event
    double max_q = 0.0;  //max collected charge per event
    double min_e = std::numeric_limits<double>::infinity();
    double max_e = 0.0;
    unsigned long long total_events = 0;  //all events
    unsigned long long collecting_events = 0;  //events with nonzero charge

    //determines range of histogram
    while(reader.Next()) {
        total_events++;
        unsigned long long event_q = 0;
        //initialize total event charge
        for(auto* p : *pixels) {
            if(p == nullptr) {
                continue;
            }
            //loop over all pixels in the event
            event_q += p->getAbsoluteCharge();
        }
        //if total charge is zero skip
        if(event_q == 0) {
            continue;
        }
        collecting_events++;
        //convert charge to energy using energy = electons * 3.9eV / 1000 keV
        const double q = static_cast<double>(event_q);
        const double e_keV = (q * ehpair_eV) * 1.0e-3;
        min_q = std::min(min_q, q);
        max_q = std::max(max_q, q);
        min_e = std::min(min_e, e_keV);
        max_e = std::max(max_e, e_keV);
    }

    const bool has_signal = std::isfinite(min_q) && max_q > 0.0 && std::isfinite(min_e) && max_e > 0.0;
    if(!has_signal) {
        std::cerr << "No collected charge entries found." << std::endl;
    }

    const double low_q = has_signal ? std::max(0.0, min_q * 0.95) : 0.0;
    const double high_q = has_signal ? max_q * 1.05 : 1.0;
    double low_e = has_signal ? std::max(0.0, min_e * 0.95) : 0.0;
    double high_e = has_signal ? max_e * 1.05 : 1.0;
    
    //define histogram ranges
    if(!(high_e > low_e)) {
        low_e = std::max(0.0, min_e - 10.0);
        high_e = max_e + 10.0;
    }

    //builds output file names
    //builds _charge_hist.csv, _energy_hist.csv, _stats.csv, _histograms.root
    const std::string input_name(filename);
    const auto dot_pos = input_name.rfind(".root");
    const std::string base_name = (dot_pos == std::string::npos) ? input_name : input_name.substr(0, dot_pos);
    const std::string charge_csv = base_name + "_charge_hist.csv";
    const std::string energy_csv = base_name + "_energy_hist.csv";
    const std::string stats_csv = base_name + "_stats.csv";
    const std::string out_hist_root = base_name + "_histograms.root";

    //creates charge histogram
    TH1D h_charge("h_deposited_charge", "Total collected charge per event;charge [e];entries", 200, low_q, high_q);
    //creates energy histogram
    TH1D h_energy("h_deposited_energy", "Total collected energy per event;energy [keV];entries", 200, low_e, high_e);

    //creates another reader and repeats the loop to fill the histograms
    TTreeReader reader2("PixelCharge", &f);
    TTreeReaderValue<std::vector<allpix::PixelCharge*>> pixels2(reader2, detector_branch.c_str());
    while(reader2.Next()) {
        unsigned long long event_q = 0;
        for(auto* p : *pixels2) {
            if(p == nullptr) {
                continue;
            }
            event_q += p->getAbsoluteCharge();
        }
        if(event_q == 0) {
            continue;
        }
        const double q = static_cast<double>(event_q);
        const double e_keV = (q * ehpair_eV) * 1.0e-3;
        h_charge.Fill(q);  //fills the charge histogram
        h_energy.Fill(e_keV);  //fills the energy histogram
    }

    //Fits the histograms with gaussian curves
    if(h_charge.GetEntries() > 0) {
        h_charge.Fit("gaus");
    }
    if(h_energy.GetEntries() > 0) {
        h_energy.Fit("gaus");
    }

    {
        //export charge histogram to csv
        std::ofstream out(charge_csv);
        out << "bin_center,bin_low,bin_high,count\n";
        //loops over the bins
        for(int i = 1; i <= h_charge.GetNbinsX(); ++i) {
            out << h_charge.GetXaxis()->GetBinCenter(i) << ","
                << h_charge.GetXaxis()->GetBinLowEdge(i) << ","
                << h_charge.GetXaxis()->GetBinUpEdge(i) << ","
                << h_charge.GetBinContent(i) << "\n";
        }
    }
    {
        //export energy histogram to csv
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
        //exports statistics
        std::ofstream out(stats_csv);
        out << "total_particles,depositing_particles,efficiency\n";
        const double eff = total_events > 0 ? static_cast<double>(collecting_events) / total_events : 0.0;
        out << total_events << "," << collecting_events << "," << eff << "\n";
        printf("MACRO_SCRIPT------>STATS_CSV containing event and particle info is %s\n", stats_csv.c_str());
    }
    {
        //saves histograms to ROOT file
        TFile out_root(out_hist_root.c_str(), "RECREATE");
        h_charge.Write();
        h_energy.Write();
        out_root.Close();
        printf("MACRO_SCRIPT---->ROOT file where histograms are saved is %s\n", out_hist_root.c_str());
    }

    //prints gaussian fit results
    if(auto* fit_q = h_charge.GetFunction("gaus")) {
        std::cout << "Collected-charge Gaussian mean = " << fit_q->GetParameter(1)
                  << " e-, sigma = " << fit_q->GetParameter(2) << " e-" << std::endl;
    }
    if(auto* fit_e = h_energy.GetFunction("gaus")) {
        std::cout << "Collected-energy Gaussian mean = " << fit_e->GetParameter(1)
                  << " keV, sigma = " << fit_e->GetParameter(2) << " keV" << std::endl;
    }
}
