# CloudDetectionPRISMA

# 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## Description

This repository contains the Python sources of the Prisma basic processing for cloud classification. Some parts for the preprocessing were adapted from the original code developed by [[1]](Vanhellemonthttps://www.sciencedirect.com/science/article/pii/S0034425718303481) - see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

# How it was trained?
<img width="2683" height="2439" alt="flujo_prisma" src="https://github.com/user-attachments/assets/965946b4-49fa-422d-bb34-e1e8a8324d62" />


# Labeled data
## Training dataset
the database is available in the following link:  
- [Training Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/training_ds.md)

## Validation dataset
the database is available in the following link:  
- [Validation Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/Validation_ds.md)
  
# Prisma program Structure:
In this project you will find:

* requirements.txt it contains all the necessary libraries;

* scripts contains a modular code:

<img width="2944" height="2078" alt="folders" src="https://github.com/user-attachments/assets/8ca17e9e-d611-4692-aa9d-9348c300f7ad" />



- trained_models: contains the best model based on optuna optimization. 
Additionaly you will find two directories. First one called [configuration_files](https://github.com/argennof/CloudDetectionPRISMA/tree/main/configuration_files), which provides examples to set the different input files to run the main classification scripts.
Second one, called [sample_validation-data](https://github.com/argennof/CloudDetectionPRISMA/tree/main/sample_validation-data) provides the access to sample prisma dataset and its vector masks.

# Prepare environment
-Example based on linux systems-

  1. Create an environment, for instance:
  ```
    $ pip install virtualenv
    $ python -m venv <virtual-environment-name>
  ```
or if necessary:
   
   ```
    $ pip3 install virtualenv
    $ python3 -m venv <virtual-environment-name>
  ```

  2. Activate your virtual environment:
  ```
      $ source <virtual-environment-name>/bin/activate
  ```
  3.  Install the requirements in the Virtual Environment, you can easily just pip install the libraries. For example:
  ```
      $ pip install numpy
  ```
  or  If you use the requirements.txt file:
  ```
      $ pip install -r requirements.txt
  ```

  4. Download the scripts available here and save them into the same directory or try via git clone source:
  ```
      $ git clone https://github.com/argennof/CloudDetectionPRISMA
  ```
Alternative you can download the zip. Please make sure to rename the directory as: ``CloudDetectionPRISMA`` after unzip the files.

  # Note:
  Given its weight, some files are attached as google drive link. Do not forget to download them:

  - [Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
  - [sample_validation_data/gpkg/](https://github.com/argennof/CloudDetectionPRISMA/blob/main/sample_validation-data/gpkg/XXXXXXXXXXXXXXXXXX) folder
  - [sample_validation_data/hdf/](https://github.com/argennof/CloudDetectionPRISMA/blob/main/sample_validation-data/hdf/XXXXXXXXXXXXXXXXXXXXXXXX) folder
  - [trained_models/knn.joblib](https://github.com/argennof/CloudDetectionPRISMA/blob/main/trained_models/XXXXXXXXXXXXXXXXXXXXXXXXXXXXX)

# Run the scripts
## To get the Cloud Mask:
  Configure the [config_getclassification.json](https://github.com/argennof/CloudDetectionPRISMA/blob/main/configuration_files/config_getclassification.json) file according to the instructions given in this [link](https://github.com//tree/main/XXXXXX).
  
  Locate at the directory called: ``main`` inside of ``CloudDetectionPRISMA`` folder. Once there, execute the next command in a terminal, for example, to run
  the classification script ([get_classification.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/get_classification.py)), you can run the following line:
  ```
      $ python get_classification.py -i <Path to the config_getclassification.json file>
  ```
## help
  Additionally, you can access to the help of each script for generate a model ([get_model.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/get_model.py)), validation ([get_validation.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/XXXXXX)) or classification ([get_classification.py]([https://github.com/XXXXXXXXX](https://github.com/argennof/CloudDetectionPRISMA/blob/main/trained_models/knn_joblib.md)) using the `-h` tag:
 
 ```
      $ python get_model.py -h
 ```

# Sample Results
Below, some sample results for Validation and Classification are shown by using the provided [models](https://github.com/argennof/CloudDetectionPRISMA/tree/main/trained_models) which were trained with the following 21 spectral bands: 

```
"bands": {
          "VNIR": [480, 559, 650, 660, 742, 762, 840, 942],
          "SWIR": [1114, 1131, 1250, 1386, 1548, 1558, 1651, 1749, 1759, 2072, 2082, 2193, 2203]
         }
```
## Validation

<img width="1582" height="765" alt="image" src="https://github.com/user-attachments/assets/fb142c36-3d4e-49f3-b6b5-439bd1516bb4" />

### Thin/Thick - Clouds
<img width="690" height="427" alt="image" src="https://github.com/user-attachments/assets/c6f7b28b-fa69-4f25-b3ab-c08dc6293b2d" />


## Classification 

<img width="1710" height="768" alt="image" src="https://github.com/user-attachments/assets/e69e4e7f-1610-46d1-9747-da4d7685bba9" />




# 📝 Authors information
This repository contains the Python sources of the Prisma basic processing and some parts were adapted from the original code developed by:
 - [x] see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

All credit goes to the corresponding author.
