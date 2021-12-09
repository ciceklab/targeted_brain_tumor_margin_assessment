# Targeted Metabolomics based Brain Tumor Margin Assessment

Source code for "Targeted Metabolomics Analyses for Brain Tumor Margin Assessment During Surgery"

## Dependencies 
The conda environment is given as **hrmas.txt** if you wish to clone the environment to your machine.
- PyTorch
- scikit-learn
- pandas
- numpy
- xlrd
- [PyNMR](https://github.com/bennomeier/pyNMR)
- [shap](https://github.com/slundberg/shap)

## Getting Started 
#Predict with your data
 - Move your dataset folder to **/predict_with_your_data/data** folder.
 - Go to **/predict_with_your_data/config.py** and change path variable and write the path to that folder. Additionally, you need to change the lib variable to path to pyNMR library.
 - If your data is aligned to acetate then you need to set **shift_** variable to 0 and **normalization_** to 1. Else, please untouch shift and normalization variables.
 - You can run **/predict_with_your_data/predict.py**, and see the predictions printed out to your console.


#Train with your data
 - Download the dataset from [here](https://zenodo.org/). Extract the compressed folder and move **dataset** folder to **/train_with_your_data/data** folder as a subdirectory.
