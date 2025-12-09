# CloudDetectionPRISMA

# Under the project - 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## Description

This repository contains the benchmark cloud mask and code for PRISMA cloud classification. 

# How it was trained?
<img width="2683" height="2439" alt="flujo_prisma" src="https://github.com/user-attachments/assets/965946b4-49fa-422d-bb34-e1e8a8324d62" />



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
Alternative you can download the zip. Please make sure to rename the directory as: ``prisma_cloud_mask`` after unzip the files.


# Run the scripts
## To get the Cloud Mask:
  Configure the [config_getclassification.json](https://github.com/argennof/CloudDetectionPRISMA/blob/main/configuration_files/config_getclassification.json) file according to the instructions given in this [link](https://github.com/argennof/CloudDetectionPRISMA/tree/main/configuration_files#config_getclassificationjson).
  
  Locate at the directory called: ``main`` inside of ``prisma_cloud_mask`` folder. Once there, execute the next command in a terminal, for example, to run
  the classification script ([get_classification.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/main/get_classification.py)), you can run the following line:
  ```
      $ python get_classification.py -i <Path to the config_getclassification.json file>
  ```

## help
  Additionally, you can access to the help of each script for generate a model ([get_model.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/main/get_model.py)), validation ([get_validation.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/main/get_validation.py)) or classification ([get_classification.py](https://github.com/argennof/CloudDetectionPRISMA/blob/main/main/get_classification.py)) using the `-h` tag:
 
 ```
      $ python get_model.py -h
 ```

# Sample of Labeled data:

![signatures_n](https://github.com/user-attachments/assets/1d5e7a20-8199-4b25-a4c8-7e4b312cf09c)

## Training dataset - labels
The database is available in the following link:  
- [Training Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/training_ds.md)

### - Classes distribution, Cloudy/Non-Cloudy and Land-cover

<img width="382" height="263" alt="image" src="https://github.com/user-attachments/assets/0ed2e30e-bf3f-42a8-a4ca-39185965d977" />

## Validation dataset  - labels
The database is available in the following link:  
- [Validation Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/Validation_ds.md)

# Sample Results
Below, some sample results for Validation and Thin/Thick are shown by using eXtreme Gradient Boosting (XGboost), k-nearest neighbors (kNN), Random Forest (RF). These were trained with the following 21 spectral bands: 

```
"bands": {
          "VNIR": [480, 559, 650, 660, 742, 762, 840, 942],
          "SWIR": [1114, 1131, 1250, 1386, 1548, 1558, 1651, 1749, 1759, 2072, 2082, 2193, 2203]
         }
```

- Example database with labels:

<img width="1038" height="333" alt="image" src="https://github.com/user-attachments/assets/89d6321f-86bf-41de-a20c-7c2f69a1848a" />


## Validation

<img width="1582" height="765" alt="image" src="https://github.com/user-attachments/assets/fb142c36-3d4e-49f3-b6b5-439bd1516bb4" />

### Thin/Thick - Clouds
<img width="690" height="427" alt="image" src="https://github.com/user-attachments/assets/c6f7b28b-fa69-4f25-b3ab-c08dc6293b2d" />

#### False Color - > Red: 1647 nm, Green: 838 nm, Blue: 482 nm


# 📝 Authors information 
This repository contains the benchmark for cloud mask - PRISMA classification. PRISMA data were available from the Italian Space Agency (ASI) website. This research was carried out in the framework of the project “SAPP4VU: Sviluppo di Algoritmi Prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation”, which was funded by the Italian Space Agency, (contract no. 2022-13-U.O., BANDO ASI DC-UOT-2019-061).
Also, this repository contains the Python sources of the Prisma basic processing and some parts were adapted from the original code developed by:
 - [x] see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 
- All credit goes to the corresponding authors.
