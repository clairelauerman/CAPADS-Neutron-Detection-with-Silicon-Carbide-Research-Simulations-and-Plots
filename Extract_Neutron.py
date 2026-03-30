# extraction script for neutrons that reads all the data from the histograms
# created in the Macro files and writes the data into a csv
import numpy as np
import ROOT


def th1_to_csv(root_file, hist_name, dir_path, out_csv):
    print(f"+++++++EXTRACT_SCRIPT+++++ROOT file being opened is {root_file}")
    print(f"++++++EXTRACT_SCRIPT++++++histogram name is {hist_name}")
    f = ROOT.TFile.Open(root_file)  # opens the root file
    d = f.Get(dir_path) if dir_path else f
    print(f"F is {f}")
    print(f"DIR_PATH IS {d}")
    if not d:
        raise RuntimeError("TDirectory not found")

    h = d.Get(hist_name)
    if not h or not h.InheritsFrom("TH1"):
        raise RuntimeError("TH1 not found")

    nb = h.GetNbinsX()  # gets the number of bins on the x-axis
    # makes a vector of all the coordinates of each bin center
    x = np.array([h.GetXaxis().GetBinCenter(i) for i in range(1, nb + 1)])
    # vector of the number of entries in each bin
    y = np.array([h.GetBinContent(i) for i in range(1, nb + 1)])
    # bin errors
    e = np.array([h.GetBinError(i) for i in range(1, nb + 1)])

    # stores all the data into a table
    data = np.column_stack((x, y, e))

    # if there is no output file
    if out_csv is None:
        out_csv = h.GetName() + ".csv"

    # saves the csv file
    np.savetxt(
        out_csv, data, delimiter=",", header="x,content,error", comments=""
    )

    f.Close()  # closes the root file
    print(
        f"+++++EXTRACT_SCRIPT+++++Wrote RUN_CSV with histogram information: {out_csv or (hist_name + '.csv')}"
    )


if __name__ == "__main__":
    import sys

    root_file = sys.argv[1]
    hist_name = sys.argv[2]
    dir_path = (
        sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "None" else ""
    )
    out_csv = sys.argv[4] if len(sys.argv) > 4 else None

    # call the function
    th1_to_csv(root_file, hist_name, dir_path, out_csv)
    # print statement confirmation
    print(
        f"+++++EXTRACT_SCRIPT+++++Wrote RUN_CSV with histogram information: {out_csv or (hist_name + '.csv')}"
    )
