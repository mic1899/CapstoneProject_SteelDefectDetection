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
|1|first-EDA.ipynb| This notebook contains the exploritory data analysis regarding this project.|
|2|convert-to-mask.ipynb| TBD |
|3|FailurePostion.ipynb| This notebook generates pixels at specified position of failure encoding from the csv file. Aftwards it plots the image with and without defect in 2 subplots to compare real pattern and defect labelled area.|
|4|BinaryCNN.ipynb| Here you can find the binary classification model as Convolutional Neural Network(CNN). This notebook tooks the data from the challenge and do binary comparisons like e.g. Class 3 Vs. Remain Classes. It is setup as to use with KERAS generators and the required folder structure. The folder structure can be created with functions in the notebook.|
|5|Clustering.ipynb| Clustering is an unsupervised machine learning task. Here in this notebook it were analysed how this technique can enhence dataset from the unlabeled images.|
|6|HOG.ipynb|Here the Histogram of Oriented Gradients were calculated from the given images. |
|7|image-augmentation.ipynb|This notebook can be used to do the image augmentation. |
|8|isolate-defectless-images.ipynb|In this notebook the images without defect labelling can be isolated.|
|9|KNN-with-augmentet-images.ipynb|Here the k-Nearest-Neighbour algorithm were used to classify the images. The images were pre-processed with augmention. Means that the give data were multiplied|
|10|KNN-with-initial.ipynb.ipynb|In here the k-Nearest-Neighbour algorithm were used to classify the images. In this case the images were not pre-processed.|
|11|Model-with-HOG-SURF.ipynb|In this notebook the images are pre-processed either with HOG method or SURF.|
|12|segmentation_multi_model.ipynb|This is the key notebook of the CNN multi model generation. Here it is used the UNET model pre-trained to efficientnetb5 database. Finally the 4 models, one for each defect class will be calculated. Afterwards you can predict with single images the 4 models response mask.|
|13|segmentation_single_model.ipynb|This notebook is similar to the "segmentation_multi_model.ipynb", at least from model perspective. Also the pre-trained UNET model is the key to predict the masks of the images. Different here it especially setup to analyze the masks of a single model of one defect class.|
|14|SURF.ipynb | To analyze the images regarding the SURF(Speeded Up Robust Features) method you can use this notebook.|
|15| unstructered-data-handling.ipnyb |TBD|



---

1 first-EDA.ipynb 
2 convert-to-mask.ipynb
3 FailurePostion.ipynb
4 BinaryCNN.ipynb
5 Clustering.ipynb
6 HOG.ipynb
7 image-augmentation.ipynb
8 isolate-defectless-images.ipynb
9 KNN-with-augmentet-images.ipynb
10 KNN-with-initial.ipynb.ipynb
11 Model-with-HOG-SURF.ipynb
12 segmentation_multi_model.ipynb
13 segmentation_single_model.ipynb
14 SURF.ipynb
15 unstructered-data-handling.ipnyb

Below you will find a brief description of what each notebook is to be used for.

### first-EDA.ipynb 
This notebook contains the exploritory data analysis regarding this project.

### 2 convert-to-mask.ipynb
TBD

### 3 FailurePostion.ipynb
This notebook generates pixels at specified position of failure encoding from the csv file.
Aftwards it plots the image with and without defect in 2 subplots to compare real pattern and defect labelled area.

### 4 BinaryCNN.ipynb
Here you can find the binary classification model as Convolutional Neural Network(CNN).
This notebook tooks the data from the challenge and do binary comparisons like e.g. Class 3 Vs. Remain Classes.
It is setup as to use with KERAS generators and the required folder structure. The folder structure can be created with functions in the notebook.

### 5 Clustering.ipynb
Clustering is an unsupervised machine learning task. 
Here in this notebook it were analysed how this technique can enhence dataset from the unlabeled images.

### 6 HOG.ipynb
Here the Histogram of Oriented Gradients were calculated from the given images.

### 7 image-augmentation.ipynb
This notebook can be used to do the image augmentation.

### 8 isolate-defectless-images.ipynb
In this notebook the images without defect labelling can be isolated.

### 9 k-NearestN-with-augmented-images.ipynb
Here the k-Nearest-Neighbour algorithm were used to classify the images.
The images were pre-processed with augmention. Means that the give data were multiplied  

### 10 k-NearestN-with-initial.ipynb.ipynb
In here the k-Nearest-Neighbour algorithm were used to classify the images. 
In this case the images were not pre-processed.

### 11 Model-with-HOG-SURF.ipynb
In this notebook the images are pre-processed either with HOG method or SURF.

### 12 segmentation_multi_model.ipynb
This is the key notebook of the CNN multi model generation. Here it is used the UNET model pre-trained to efficientnetb5 database. Finally the 4 models, one for each defect class will be calculated.
Afterwards you can predict with single images the 4 models response mask.

### 13 segmentation_single_model.ipynb
This notebook is similar to the "segmentation_multi_model.ipynb", at least from model perspective. Also the pre-trained UNET model is the key to predict the masks of the images. Different here it especially setup to analyze the masks of a single model of one defect class. 

### 14 SURF.ipynb
To analyze the images regarding the SURF(Speeded Up Robust Features) method you can use this notebook. 

### 15 unstructered-data-handling.ipnyb
TBD


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

 