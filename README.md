# Targeted Metabolomics based Brain Tumor Margin Assessment

Source code for "Targeted Metabolomics Analyses for Brain Tumor Margin Assessment During Surgery"

## Dependencies 
The conda environment is given as **hrmas.yml** if you wish to clone the environment to your machine. Please consider using our environment to avoid any inconvenience resulted from version differences.
- PyTorch
- scikit-learn
- pandas
- numpy
- xlrd
- [PyNMR](https://github.com/bennomeier/pyNMR)
- [shap](https://github.com/slundberg/shap)

# Getting Started 
## Predict with your data
 - Go to **/predict_with_your_data/config.py** and change path variable and write the path to that folder and change the lib variable to the path to pyNMR library.
 - Please run the code with the anonymous sample and see the prediction printed out.
 - Move your dataset folder to **/predict_with_your_data/data** folder.
 - If your data is aligned to lactate and normalized to acetate then please keep **shift_** and **normalization_** variables untouched. Else, if the alignment and normalization of your data is different than that, please change **shift_** and **normalization_** variables accordingly. We recommend -1515 and 1.6 respectively, if your data is aligned to water.
 - You can run **/predict_with_your_data/predict.py**, and see the predictions printed out to your console.
 - The script will predict whether the samples in your dataset are of control, benign glioma or aggressive glioma tissue. You will see the predictions with their corresponding probability.


There are 3 main trainings you may do in this repository. These are **automated metabolite quantification**, **pathologic classification** and **sample visualization**.

## Train with our dataset
 - Download the dataset from [here](https://zenodo.org/record/5774947). Extract the compressed folder and move all the contents of **cpmg_dataset** folder to **/train_with_your_data/data/cpmg** folder and move all the contents of **eretic_dataset** folder **/train_with_your_data/data/eretic_cpmg** as a subdirectory.
 - Download the **Supplementery Table 2** and rename it to **Table_S2.xls**, then move it into **data_xlsx** folder.
 - Go to **scripts/config_ult.py** and change the base variable and write the path to that folder and change the lib variable to the path to pyNMR library.
 - Run **create_processed_cpmg_dataset.py** and **create_processed_eretic_cpmg_dataset.py** scripts.
 - **pathologic classification** training will require models of **automated metabolite quantification**. Go to the **automated metabolite quantification** folder and train all the models for all metabolites by running **train.py** script.
 - Go to **pathologic classification** and train both the **benign vs aggressive** and **control vs tumor** classification models by **train.py** script under the model of interest. We recommend you to use the **/cpmg/pathologic_classification/control_tumor/metabolites/predicted/all_metabolites/RF** and **/cpmg/pathologic_classification/benign_aggressive/metabolites/predicted/all_metabolites/RF**.


## Train with your data
 - If your dataset consists of CPMG samples, move the contents of that dataset into **/train_with_your_data/data/cpmg** folder, likewise if your data consists of ERETIC-CPMG samples, move the contents of that dataset into **/train_with_your_data/data/eretic_cpmg** folder.
 - The model training requires an **xls** file, and we release our file in **Supplementery Table 2**. Please format your spreadsheet according to our Table.
 - Go to **scripts/config_ult.py** and change the base variable and write the path to that folder and change the lib variable to the path to pyNMR library.
 - Run **create_processed_cpmg_dataset.py** and **create_processed_eretic_cpmg_dataset.py** scripts.
 - Apply the steps 4 and 5 of **Train with our dataset**.
 

 
