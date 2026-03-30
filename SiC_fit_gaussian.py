# fits the data in the csv files into a guassian distribution and calculates statistics
import sys

import numpy as np
from scipy.optimize import curve_fit


# standalone guassian function
def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# read csv file and create histogram
def fit_histogram_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    if {"x", "content"}.issubset(data.dtype.names):
        x = data["x"]  # bin centers
        y = data["content"]  # counts
        # errors
        yerr = (
            data["error"]
            if "error" in data.dtype.names
            else np.sqrt(np.maximum(y, 1.0))
        )
    elif {"bin_center", "count"}.issubset(data.dtype.names):
        x = data["bin_center"]
        y = data["count"]
        yerr = np.sqrt(np.maximum(y, 1.0))
    else:
        raise RuntimeError(
            "Unexpected CSV headers. Expected x/content/error or bin_center/count."
        )

    if len(x) == 0:
        raise RuntimeError(f"No bins found in CSV: {csv_path}")

    # amount of entries with each energy
    total_entries = float(np.sum(y))
    if total_entries <= 0:
        raise RuntimeError(
            f"Histogram has zero total entries: {csv_path}. "
            "Check histogram range and upstream simulation output."
        )

    # Restrict fit inputs to sufficiently populated bins for stability
    mask = y >= 3
    xf = x[mask]
    yf = y[mask]
    yerrf = yerr[mask]
    if len(xf) < 4:
        raise RuntimeError(
            f"Not enough populated bins ({len(xf)}) for Gaussian fit in {csv_path}"
        )

    # Initial guesses
    a0 = np.max(yf)
    mu0 = xf[np.argmax(yf)]
    mu_w = np.sum(xf * yf) / np.sum(yf)
    sigma0 = np.sqrt(np.sum(yf * (xf - mu_w) ** 2) / np.sum(yf))
    sigma0 = max(float(sigma0), 1e-6)

    x_min = float(np.min(xf))
    x_max = float(np.max(xf))
    sigma_upper = max((x_max - x_min), sigma0 * 10.0, 1.0)

    # First pass fit
    popt1, pcov1 = curve_fit(
        gaussian,
        xf,
        yf,
        p0=[a0, mu0, sigma0],
        sigma=yerrf,
        absolute_sigma=True,
        bounds=([0.0, x_min, 1e-6], [np.inf, x_max, sigma_upper]),
        maxfev=20000,
    )

    # Second pass fit using first-pass mean/sigma results
    mu1 = float(popt1[1])
    sigma1 = abs(float(popt1[2]))
    window = 2.5 * max(sigma1, 1e-6)
    mask2 = (xf >= mu1 - window) & (xf <= mu1 + window)
    if np.sum(mask2) >= 4:
        popt, pcov = curve_fit(
            gaussian,
            xf[mask2],
            yf[mask2],
            p0=[float(np.max(yf[mask2])), mu1, max(sigma1, 1e-6)],
            sigma=yerrf[mask2],
            absolute_sigma=True,
            bounds=([0.0, x_min, 1e-6], [np.inf, x_max, sigma_upper]),
            maxfev=20000,
        )
    else:
        popt, pcov = popt1, pcov1

    if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(pcov)):
        raise RuntimeError(f"Non-finite fit result for {csv_path}")

    perr = np.sqrt(np.diag(pcov))

    a, mu, sigma = popt
    a_err, mu_err, sigma_err = perr
    if sigma <= 0:
        raise RuntimeError(f"Non-physical sigma <= 0 for {csv_path}")
    if not np.isfinite(sigma_err):
        raise RuntimeError(f"Non-finite sigma error for {csv_path}")
    if sigma_err / sigma > 0.5:
        raise RuntimeError(
            f"Unstable sigma fit (relative error {sigma_err / sigma:.2f}) for {csv_path}"
        )

    total_entries_err = np.sqrt(np.sum(yerr**2))

    return {
        "amplitude": float(a),
        "amplitude_error": float(a_err),
        "mean": float(mu),
        "mean_error": float(mu_err),
        "sigma": float(sigma),
        "sigma_error": float(sigma_err),
        "total_entries": float(total_entries),
        "total_entries_error": float(total_entries_err),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python fit_gaussian.py <hist_csv>")
        print(
            "CSV format: bin_center,bin_low,bin_high,count OR x,content,error"
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    fit = fit_histogram_csv(csv_path)

    print(f"mean = {fit['mean']:.6g}")
    print(f"mean_error = {fit['mean_error']:.6g}")
    print(f"sigma = {fit['sigma']:.6g}")
    print(f"sigma_error = {fit['sigma_error']:.6g}")
    print(f"total_entries = {fit['total_entries']:.6g}")
    print(f"total_entries_error = {fit['total_entries_error']:.6g}")


if __name__ == "__main__":
    main()
