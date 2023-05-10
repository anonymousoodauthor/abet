# Object Detection using Ablated Learned Temperature Energy (AbeT)

This folder contains code for model training and evaluation using the AbeT method in object detection. Specifically we
modify code from the baseline SOTA [VOS method repo](https://github.com/deeplearning-wisc/vos).

# Usage

## Requirements
```
pip install -r requirements.txt
```
In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset prep
In order to perform model training and evaluation the PASCAL VOC and COCO datasets need to be downloaded. Download instructions
are copied from the original [VOS repo](https://github.com/deeplearning-wisc/vos):
**PASCAL VOC**

Download the processed VOC 2007 and 2012 dataset from [here](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         └── val_coco_format.json

**COCO**

Download COCO2017 dataset from the [official website](https://cocodataset.org/#home). 

Download the OOD dataset (json file) when the in-distribution dataset is Pascal VOC from [here](https://drive.google.com/file/d/1Wsg9yBcrTt2UlgBcf7lMKCw19fPXpESF/view?usp=sharing). 

Download the OOD dataset (json file) when the in-distribution dataset is BDD-100k from [here](https://drive.google.com/file/d/1AOYAJC5Z5NzrLl5IIJbZD4bbrZpo0XPh/view?usp=sharing).

Put the two processed OOD json files to ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_ood_wrt_bdd_rm_overlap.json
            └── instances_val2017_ood_rm_overlap.json
         ├── train2017
         └── val2017


## Model Training
Model training code is run from the `detection` folder.
```
python train.py 
--dataset-dir path/to/dataset/dir
--num-gpus 1 
--config-file configs/abe_t.yaml 
--random-seed 0 
--resume
```

## Model Evaluation
Evaluation is run in 3 steps: evaluating in-distribution dataset,
evaluation out of distribution dataset, and then obtaining final FP@95, AUROC, and AUPRC metrics.

We provide our trained model weights for download [here](https://drive.google.com/file/d/1bsO2ZaDdu9fwFK7dQ_tHFipfW2TA-ZXQ/view?usp=share_link).
These weights can directly be loaded into evaluation using the `--previous_model_weights` argument in `apply_net.py`. Otherwise,
the seed from training will be used to find the model weight path.

In distribution dataset run:
```bash
python apply_net.py 
--dataset-dir path/to/vos/dataset/dir
--test-dataset voc_custom_val 
--config-file abe_t.yaml 
--inference-config standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
--previous_model_weights /path/to/abet_fasterrcnn_voc.pth
```

OOD dataset run:
```bash
python apply_net.py
--dataset-dir path/to/coco/dataset/dir
--test-dataset coco_ood_val 
--config-file abe_t.yaml 
--inference-config standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
--previous_model_weights /path/to/abet_fasterrcnn_voc.pth
```

Final metrics:
```bash
python voc_coco_eval.py 
--seed 0
```

