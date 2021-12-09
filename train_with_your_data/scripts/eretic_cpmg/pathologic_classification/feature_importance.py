import numpy as np
import shap
import pdb
import matplotlib.pyplot as plt
''' SHAP value based feature importance analysis and plotting feature importances '''


# conversion from spectrum index (between 0 and 16313) to ppm (between -2 ppm to 12 ppm).
RAW_SPECTRUM_LENGTH = 16314
MIN_PPM = -2
MAX_PPM = 12
def find_ppm_value(idx):
    return round((14 * (idx) / RAW_SPECTRUM_LENGTH -2), 2)

# plot all shap values associated with a dataset and spectrum of length 16314.
def plot_all_shap_spectrum(shap_values, spectrum):
    # normalize each ppm 
    spectrum = spectrum / np.amax(spectrum,axis=0,keepdims=True)
    # prepare scatter plot entries
    xs = []
    ys = []
    vals = []
    sizes = []
    for i in range(shap_values.shape[1]):
        for j in range(shap_values.shape[0]):
            xs.append((i+1))
            ys.append(shap_values[j,i])
            vals.append(spectrum[j,i])
            sizes.append(1)
    # scatter plot
    fig = plt.figure()
    res = plt.scatter(xs,ys,c=vals,s=sizes,marker='o',cmap="cool",alpha=0.3)
    # colorbar at right of the figure
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Signal Amplitude', rotation=90)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
        va='top', ha='center')
    cbar.ax.text(0.5, 1.01, 'High', transform=cbar.ax.transAxes, 
        va='bottom', ha='center')
    # change x axis scale 
    locs, labels = plt.xticks()
    locs = np.arange(-2000, 14501, 500)
    labels = [find_ppm_value(float(item)) for item in locs]
    plt.xticks(locs, labels,rotation = (45), fontsize = 10, va='top', ha='center')
    # set axis labels
    plt.xlabel("ppm")
    plt.ylabel("SHAP Value")
    plt.tight_layout()
    return fig

# sort shap values in descending order and return corresponding indices in terms of both spectrum scale and ppm scale. 
def sort_shap_values(shap_values):
    abs_shap_values = np.absolute(shap_values)
    max_abs_shap_values = np.amax(abs_shap_values, axis=0)
    top_k_ind  = max_abs_shap_values.argsort()[(-1)*shap_values.shape[1]:][::-1]
    temp = [find_ppm_value(x) for x in top_k_ind]
    temp.sort()
    return top_k_ind, temp
