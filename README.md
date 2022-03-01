# CapstoneProject Steel-Defect-Detection
This work shown here is about the capstone project of a group of the data science course 2022 from neuefische GmbH.

Down below you find the required environment to execute the workbooks.

If you have further questions to this work here do not hesitate to contact us.



## Task
The project is based on the Serverstal-Steel-Defect-Detection published on the kaggle.com site.

Please check the website for details to the challenge: https://www.kaggle.com/c/severstal-steel-defect-detection

The data set was loaded from there and processed analogously to the task description of the challenge.

It is not uploaded here!

This project here was limited to a processing time of 4 weeks only. 


## Objectives
According the challenge one objective is the Sørensen-Dice metric(https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient). 
In easy words that means that predicted masks should fit pixelwise very best to the orign mask.

Here in our task we set a specific focus to help the stakeholder to improve their process and product quality. 




## Data 
Input data were taken from https://www.kaggle.com/c/severstal-steel-defect-detection/data

The data itself contains images of steels taken after the production process


## Overview Notebooks
As you can see here an overview about the generated notebooks during this project.



---
##  __Notebooks__

|No.|Notebook| Brief Summary|
|---|---|---|
|0|00_data_preparations.ipynb|This notebook is to extract the information from \*.csv to use for different workbooks here .|
|1|classification_HOG_k-Nearest_Neighbor.ipynb|Here the Histogram of Oriented Gradients were calculated from the given images and applied to a k-NN algorithm|
|2|classification-initial-and-augmented-images-k-nearest-neighbor.ipynb|Classification notebook with SURF method appied to the k-NN algorithm|
|3|classification-initial-and-augmented-images-k-nearest-neighbor.ipynb|Classification notebook with the initial images and with the augemented images algorithim|
|4|clustering.ipynb|Clustering is an unsupervised machine learning task. Here in this notebook it were analysed how this technique can enhence dataset from the unlabeled images.|
|5|convert-to-mask.ipynb|This notebook generates masks (matrix of zeros and ones - ones mark defected areas) from the encoded pixels |
|6|defectPostion.ipynb|This notebook generates pixels at specified position of defect encoding from the csv file. Aftwards it plots the image with and without defect in 2 subplots to compare real pattern and defect labelled area.|
|7|EDA.ipynb|This notebook contains the exploritory data analysis regarding this project.|
|8|image-augmentation.ipynb|This notebook can be used to do the image augmentation. |
|9|isolate-defectless-images.ipynb|In this notebook the images without defect labelling can be isolated.|
|10|segmentation_multi_model.ipynb|This is the key notebook of the CNN multi model generation. Here it is used the UNET model pre-trained to efficientnetb5 database. Finally the 4 models, one for each defect class will be calculated. Afterwards you can predict with single images the 4 models response mask.|
|11|segmentation_single_model.ipynb|This notebook is similar to the "segmentation_multi_model.ipynb", at least from model perspective. Also the pre-trained UNET model is the key to predict the masks of the images. Different here it especially setup to analyze the masks of a single model of one defect class.|
|12|unstructered-data-handling.ipnyb |This notebook generates the \*.csv file for all images and their the storage path|


---

## Environment
Make sure you have the latest version of macOS (currently Monterey) installed.
Also make sure that xcode is installed and updated: 

```BASH
xcode-select --install
```

Then we can go on to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.1

pip install -U pip
pip install --no-binary=h5py h5py
pip install tensorflow-macos
pip install tensorflow-metal
pip install -r requirements.txt
pip install opencv-python
pip install sklearn
```

 