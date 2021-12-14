# Targeted Metabolomics based Brain Tumor Margin Assessment

Source code for "Targeted Metabolomics Analyses for Brain Tumor Margin Assessment During Surgery"

## Dependencies 
The conda environment is given as **hrmas.yml** if you wish to clone the environment to your machine.
- PyTorch
- scikit-learn
- pandas
- numpy
- xlrd
- [PyNMR](https://github.com/bennomeier/pyNMR)
- [shap](https://github.com/slundberg/shap)

## Getting Started 
# Predict with your data
 - Go to **/predict_with_your_data/config.py** and change path variable and write the path to that folder and change the lib variable to the path to pyNMR library.
 - Please run the code with the anonymous sample and see the prediction printed out.
 - Move your dataset folder to **/predict_with_your_data/data** folder.
 - If your data is aligned to lactate and normalized to acetate then please keep **shift_** and **normalization_** variables untouched. Else, if the alignment and normalization of your data is different than that, please change **shift_** and **normalization_** variables accordingly. We recommend -1515 and 1.6 respectively, if your data is aligned to water.
 - You can run **/predict_with_your_data/predict.py**, and see the predictions printed out to your console.


# Train with your data
 - Download the dataset from [here](https://zenodo.org/record/5774947). Extract the compressed folder and move all the contents of **cpmg_dataset** folder to **/train_with_your_data/data/cpmg** folder and move all the contents of **eretic_dataset** folder **/train_with_your_data/data/eretic_cpmg** as a subdirectory.
 - There are 3 main trainings you may do in this repository. These are **automated metabolite quantification**, **pathologic classification** and **sample visualization**.

 
