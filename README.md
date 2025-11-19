# CloudDetectionPRISMA

# Under the project - 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## Description

This repository contains the benchmark cloud mask for PRISMA cloud classification. 

# How it was trained?
<img width="2683" height="2439" alt="flujo_prisma" src="https://github.com/user-attachments/assets/965946b4-49fa-422d-bb34-e1e8a8324d62" />


# Sample of Labeled data:

![signatures_n](https://github.com/user-attachments/assets/1d5e7a20-8199-4b25-a4c8-7e4b312cf09c)


## Training dataset
The database is available in the following link:  
- [Training Database](https://github.com/argennof/CloudDetectionPRISMA/blob/main/data/training_ds.md)

## Validation dataset
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

## Validation

<img width="1582" height="765" alt="image" src="https://github.com/user-attachments/assets/fb142c36-3d4e-49f3-b6b5-439bd1516bb4" />

### Thin/Thick - Clouds
<img width="690" height="427" alt="image" src="https://github.com/user-attachments/assets/c6f7b28b-fa69-4f25-b3ab-c08dc6293b2d" />

False Color - > Red: 1647 nm, Green: 838 nm, Blue: 482 nm


# 📝 Authors information 
- This repository contains the benchmark for cloud mask - PRISMA classification. PRISMA data were available from the Italian Space Agency (ASI) website. This research was carried out in the framework of the project “SAPP4VU: Sviluppo di Algoritmi Prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation”, which was funded by the Italian Space Agency, (contract no. 2022-13-U.O., BANDO ASI DC-UOT-2019-061). All credit goes to the corresponding authors.
