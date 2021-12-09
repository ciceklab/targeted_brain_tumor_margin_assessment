import numpy as np 
import pdb
import matplotlib.pyplot as plt
import os

def plot_loss(train_losses, valid_losses, test_idx, savepath, metabolite_names, model_name):
    fig, axs = plt.subplots(1, len(metabolite_names), figsize=(3*len(metabolite_names), 3), sharey=False, sharex=True)
    for idx, metabolite in enumerate(metabolite_names):
        axs[idx].set_xlabel(metabolite)
        axs[idx].plot(train_losses[metabolite], label="Training")
        axs[idx].plot(valid_losses[metabolite], label="Validation")
    axs[0].set_ylabel("Loss")   
    plt.suptitle(f"Loss Plot of Test Fold {test_idx} for {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"test_fold_{test_idx}.pdf"))
    plt.close()