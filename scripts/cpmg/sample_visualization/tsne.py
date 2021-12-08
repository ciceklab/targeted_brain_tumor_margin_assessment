import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys 
import pdb
import os
from load_fully_quantified_predicted_cpmg_data import fq_v_ppm_spectra, fq_v_spectra, fq_v_statistics, fq_v_quant, fq_v_class_labels, fq_v_metabolite_names, fq_v_fold_dct, fq_v_pred_quant, SEED

plot_base_path = f"../plots/seed_{SEED}"

# find samples belonging to control, benign and aggressive classes 
control_idx = np.where(fq_v_class_labels == 2)[0].tolist()
benign_idx = np.where(fq_v_class_labels == 0)[0].tolist()
aggressive_idx = np.where(fq_v_class_labels == 1)[0].tolist()

'''
apply tsne to raw spectrum
'''
raw_spectrum_tsne = TSNE(n_components=2, n_jobs=-1, random_state=SEED, init="random", perplexity=20, method="exact", learning_rate=200)
transformed_spectra = raw_spectrum_tsne.fit_transform(fq_v_spectra)
# plot tsne (control and tumor) 
plt.figure(figsize=(5,5))
plt.title("Control and Tumor Samples (TSNE on raw spectrum)")
plt.scatter(transformed_spectra[benign_idx+aggressive_idx][:,0], transformed_spectra[benign_idx+aggressive_idx][:,1], c="b", label="Tumor")
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="g", label="Control")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ct.tsne.raw_spectrum.pdf"))
plt.close()
# plot tsne (benign and aggressive) 
plt.figure(figsize=(5,5))
plt.title("Benign and Aggressive Tumor CPMG Samples (TSNE on raw spectrum)")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="c", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="m", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ba.tsne.raw_spectrum.pdf"))
plt.close()
# plot tsne (control, benign and aggressive) 
plt.figure(figsize=(5,5))
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="r", label="Control")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="y", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="g", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_base_path, "cpmg.cba.tsne.raw_spectrum.pdf"))
plt.close()

'''
apply tsne to uncharacterised region
'''
raw_spectrum_tsne = TSNE(n_components=2, n_jobs=-1, random_state=SEED, init="random", perplexity=20, method="exact", learning_rate=200)
region_start = 11613
region_end = 11755
transformed_spectra = raw_spectrum_tsne.fit_transform(fq_v_spectra[:,region_start:(region_end+1)])
# plot tsne (control and tumor) 
plt.figure(figsize=(5,5))
plt.title("Control and Tumor CPMG Samples (TSNE on uncharacterised region)")
plt.scatter(transformed_spectra[benign_idx+aggressive_idx][:,0], transformed_spectra[benign_idx+aggressive_idx][:,1], c="b", label="Tumor")
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="g", label="Control")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ct.tsne.uncharacterised_region.pdf"))
plt.close()
# plot tsne (benign and aggressive) 
plt.figure(figsize=(5,5))
plt.title("Benign and Aggressive Tumor Samples (TSNE on uncharacterised region)")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="c", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="m", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ba.tsne.uncharacterised_region.pdf"))
plt.close()
# plot tsne (control, benign and aggressive) 
plt.figure(figsize=(5,5))
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="r", label="Control")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="y", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="g", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_base_path, "cpmg.cba.tsne.uncharacterised_region.pdf"))
plt.close()

'''
apply tsne to ppm spectrum
'''
raw_spectrum_tsne = TSNE(n_components=2, n_jobs=-1, random_state=SEED, init="random", perplexity=20, method="exact", learning_rate=200)
transformed_spectra = raw_spectrum_tsne.fit_transform(fq_v_ppm_spectra)
# plot tsne (control and tumor) 
plt.figure(figsize=(5,5))
plt.title("Control and Tumor CPMG Samples (TSNE on ppm spectrum)")
plt.scatter(transformed_spectra[benign_idx+aggressive_idx][:,0], transformed_spectra[benign_idx+aggressive_idx][:,1], c="b", label="Tumor")
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="g", label="Control")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ct.tsne.ppm_spectrum.pdf"))
plt.close()
# plot tsne (benign and aggressive) 
plt.figure(figsize=(5,5))
plt.title("Benign and Aggressive Tumor Samples (TSNE on ppm spectrum)")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="c", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="m", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ba.tsne.ppm_spectrum.pdf"))
plt.close()
# plot tsne (control, benign and aggressive) 
plt.figure(figsize=(5,5))
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="r", label="Control")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="y", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="g", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_base_path, "cpmg.cba.tsne.ppm_spectrum.pdf"))
plt.close()

'''
apply tsne to ground truth metabolite quantifications
'''
raw_spectrum_tsne = TSNE(n_components=2, n_jobs=-1, random_state=SEED, init="random", perplexity=20, method="exact", learning_rate=200)
transformed_spectra = raw_spectrum_tsne.fit_transform(fq_v_quant)
# plot tsne (control and tumor) 
plt.figure(figsize=(5,5))
plt.title("Control and Tumor CPMG Samples (TSNE on ground truth metabolites)")
plt.scatter(transformed_spectra[benign_idx+aggressive_idx][:,0], transformed_spectra[benign_idx+aggressive_idx][:,1], c="b", label="Tumor")
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="g", label="Control")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ct.tsne.gt_quant.pdf"))
plt.close()
# plot tsne (benign and aggressive) 
plt.figure(figsize=(5,5))
plt.title("Benign and Aggressive Tumor Samples (TSNE on ground truth metabolites)")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="c", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="m", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ba.tsne.gt_quant.pdf"))
plt.close()
# plot tsne (control, benign and aggressive) 
plt.figure(figsize=(5,5))
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="r", label="Control")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="y", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="g", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_base_path, "cpmg.cba.tsne.gt_quant.pdf"))
plt.close()

'''
apply tsne to predicted metabolite quantifications
'''
raw_spectrum_tsne = TSNE(n_components=2, n_jobs=-1, random_state=SEED, init="random", perplexity=20, method="exact", learning_rate=200)
transformed_spectra = raw_spectrum_tsne.fit_transform(fq_v_pred_quant)
# plot tsne (control and tumor) 
plt.figure(figsize=(5,5))
plt.title("Control and Tumor CPMG Samples (TSNE on predicted metabolites)")
plt.scatter(transformed_spectra[benign_idx+aggressive_idx][:,0], transformed_spectra[benign_idx+aggressive_idx][:,1], c="b", label="Tumor")
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="g", label="Control")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ct.tsne.pred_quant.pdf"))
plt.close()
# plot tsne (benign and aggressive) 
plt.figure(figsize=(5,5))
plt.title("Benign and Aggressive Tumor Samples (TSNE on predicted metabolites)")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="c", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="m", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

plt.savefig(os.path.join(plot_base_path, "cpmg.ba.tsne.pred_quant.pdf"))
plt.close()
# plot tsne (control, benign and aggressive) 
plt.figure(figsize=(5,5))
plt.scatter(transformed_spectra[control_idx][:,0], transformed_spectra[control_idx][:,1], c="r", label="Control")
plt.scatter(transformed_spectra[benign_idx][:,0], transformed_spectra[benign_idx][:,1], c="y", label="Benign")
plt.scatter(transformed_spectra[aggressive_idx][:,0], transformed_spectra[aggressive_idx][:,1], c="g", label="Aggressive")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_base_path, "cpmg.cba.tsne.pred_quant.pdf"))
plt.close()

# ppm spectrum
pdb.set_trace()