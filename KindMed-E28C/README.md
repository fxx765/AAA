# KindMed: Knowledge-Induced Medicine Prescribing Network for Medication Recommendation

This repository provides the implementation of KindMed as the medical knowledge graph-driven medicine recommender framework.

## Dependencies
Our implementation mainly utilized PyTorch 1.11.0 and PyTorch Geometric 2.2.0. Additional tensorflow/tensorboard libraries was used for training logging.
```
torch==1.11.0
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
torch-geometric==2.2.0
tensorboard==2.11.2
tensorflow-gpu==2.11.0
```

## Training
For training KindMed, run the following code:
```
python main_kindmed.py --gpu_id=0 --phase='training'
```

## Evaluation
For testing the KindMed, provide the optimized model on 'path_to_model' variable, and run the following code:
```
python main_kindmed.py --gpu_id=0 --phase='testing'
```

### Code Details
- main_kindmed.py : the main code to execute the proposed model for training / testing based on the given argument(s).
- models.py : the code that defines the model of KindMed and the fusion module
- losses.py : the code that defines the DDI loss to train the model
- helpers.py : the code that defines a set of helper functions for running the model

Apart from that, we provide additional data as follows:
- data/output_kg/* : a set of pre-processed data extracted from the MIMIC-III EHR database augmented with additional external knowledge
- log/KindMed/KindMed_optimized : a folder containing an optimized model of KindMed with a DDI threshold of 0.08 (reported in the main articles)
