# Semantic Segmentation using Ablated Learned Temperature Energy (AbeT)

This folder contains code for model training and evaluation using the AbeT method in semantic segmentation. Specifically, we modify code from two repositories: [NVIDIA Semantic Segmentation SDCNET](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) for model training and [Meta-OOD](https://github.com/robin-chan/meta-ood) for model evaluation.

We focus our experiments on the DeepLabV3+WideResNet38 model architecture, with pre-training on [Mapillary Vistas](https://www.mapillary.com/dataset/vistas/) and finetuning on [Cityscapes](https://www.cityscapes-dataset.com/), which are treated as the in-distribution datasets. The models are evaluated against [LostAndFound](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) and [RoadAnomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/), which are treated as the out-of-distribution datasets.


# Usage

## 1. Model Training

We conduct model training in the `AbeT/segmentation/nvidia-semseg-training` subfolder, which is a modified version of the [NVIDIA Semantic Segmentation SDCNET](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) repository. Follow the environment setup instructions in the README.

### 1a. Dataset Preparation
Follow the instructions in `PREPARE_DATASETS.md` to download and construct the data for both the Mapillary Vistas and Cityscapes datasets in the `AbeT/segmentation/datasets` folder. For Mapillary, this is as easy as creating an account, downloading, and unzipping the dataset: `mapillary-vistas-dataset_public_v1.1.zip`. For Cityscapes, you must create an account then follow the instructions in this [article](https://towardsdatascience.com/download-city-scapes-dataset-with-script-3061f87b20d7) to download the dataset. Finally, ensure the directory structures within `segmentation/datasets` matches the structure in `PREPARE_DATASETS.md`

### 1b. Training Notes
Both pretraining and finetuning are done through the `main()` function in `AbeT/segmentation/nvidia-semseg-training/train.py`, which is called by the set of scripts in `AbeT/segmentation/nevidia-semseg-training/scripts` folder. We modify the Mapillary pretrain and Cityscapes finetune for WideResNet38 scripts. Note that the `--apex` and `--syncbn` arguments now correspond to new Torch equivalents to the old `apex` library, specifically `torch.nn.parallel.DistributedDataParallel` and `torch.nn.SyncBatchNorm` respectively. Additionally, note that the `--temperature_model learned` command creates a model with the learned temperature layer and cosine logit head, as well as the `--fp16` flag to train in half precision. Models are trained on 8 GPUs with 8 distributed processes by default; change these numbers to fit the number of GPUs on your machine.

### 1c. Mapillary Pretrain
Once the Mapillary dataset has been downloaded, we can pretrain a model from the `AbeT/segmentation/nvidia-semseg-training` folder using: 
```
./scripts/train_mapillary_WideResNet38.sh
```
This executes the training script to pretrain a DeepWV3Plus model on Mapillary, with a number of very specific modifiers. Following the original [NVIDIA Semantic Segmentation SDCNET](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) instructions, once mIOU has reached 0.5+, the model is considered pretrained. At this point, move the model to the `AbeT/segmentation/weights` folder and begin the next stage of training.


### 1d. Cityscapes Finetune
Modify the Cityscapes finetuning script such that the `--snapshot` argument points to your pretrained Mapillary model. Execute the Cityscapes finetuning script from the `AbeT/segmentation/nvidia-semseg-training` folder using: 
```
./scripts/train_cityscapes_WideResNet38.sh
```
This trains the snapshot model on the Cityscapes dataset. Following the original [NVIDIA Semantic Segmentation SDCNET](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) instructions, once mIOU has reached 0.8+, training is considered finished. At this point, move the model to the `AbeT/segmentation/weights` folder and begin OOD evaluation.


## 2. Model Evaluation
We cnduct model evaluation in the `AbeT/segmentation/meta-ood-evaluation` subfolder, which contains a modified version of the [Meta-OOD](https://github.com/robin-chan/meta-ood) repository. Follow the instructions for environment setup in the README. For reproducibility purposes, we host our semantic segmentation models in a [a common Google Drive folder](https://drive.google.com/drive/folders/1foWuTJX_JmiGF7vPRTxlui4TMr7fa9EJ?usp=sharing). Our Learned Temperature with Cosine Logit Head semantic segmentation model finetuned on Cityscapes can be downloaded [here](https://drive.google.com/drive/folders/1F8fuOts74TdpLZfpK4ZNNpgXV19uLg0T?usp=share_link), as well as the Mapillary pretrained model [here](https://drive.google.com/drive/folders/11bG38bFFZpoGfwuNr2Dhzx7k7GPxHx78?usp=share_link). To reproduce our results, place the Cityscapes finetune model in the `AbeT/segmentation/weights` folder, place the path in `AbeT/segmentation/meta-ood-evaluation/scripts/run_evaluation_with_args.sh`, and follow the instructions below.

### 2a. Dataset Preperation
We download and prepare two OOD datasets for model evaluation. First, download the LostAndFound dataset [here](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) and the RoadAnomaly dataset from [here](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)(provided by the [SynBoost paper](https://github.com/giandbt/synboost)) to the `AbeT/segmentation/datasets` folder. Unzip these datasets and make sure their directory structure matches the layout in `AbeT/segmentation/meta-ood-evaluation/src/dataset/cityscapes.py` and `AbeT/segmentation/meta-ood-evaluation/src/dataset/road_anomaly.py` respectively.

### 2b. OOD Evaluation
To evaluate a model on one of the OOD test sets, modify `AbeT/segmentation/meta-ood-evaluation/scripts/run_evaluation_with_args.sh` to the name and split of the dataset, model specifics and path, and finally the scoring function to use. This calls the `main()` function in `AbeT/segmentation/meta-ood-evaluation/evaluation.py`. This script can support running other models and other scoring functions like entropy or max softmax probability; to see the full list of scoring functions and their specified keys, see `AbeT/segmentation/meta-ood-evaluation/src/scoring_fns.py`. For evaluating our model, set `--temperature_model leaned` and `--score_function abet` and provide a path to the trained temperature model. Run the evaluation from the `AbeT/segmentation/meta-ood-evaluation` subfolder using:
```
./scripts/run_evaluation_with_args.sh
```

### 2c. Metrics and Visualization
After running a model evaluation, we can examine the model's metrics and visualize the score distribution using `AbeT/segmentation/meta-ood-evaluation/results_visualization.ipynb`. We reccomend evaluating the learned temperature model as well as other pre-trained non-temperature models with other various scoring functions for a richer comparison of metrics and visualizations. Copy and update the base arguments with each model you want to compare, then run the notebook to visualize metrics and create the matrix of images masked by the OOD score distribution.
