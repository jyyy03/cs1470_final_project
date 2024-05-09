# [Team MINOCA] cs1470_final_project

The project is based on this paper: Adversarial Learning for Semi-Supervised Semantic Segmentation: https://arxiv.org/pdf/1802.07934.pdf

Source Code: https://github.com/hfslyc/AdvSemiSeg/tree/master

Data: https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation 

## Required Pre-trained Model and Dataset

Pre-trained model's checkpoints are loaded from DrSleep's [google drive](https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU?resourcekey=0-9Ui2e1br1d6jymsI6UdGUQ). Please download the files and put them under `model/pretrained`.

Please put downloaded data into this directory: `dataset/Forest`

## How to run

`python3 train.py`

You should be able to see a pop-up window of the last image from the testing result, and a loss and metric plot `loss_metric.png` in the main folder.

## Hyperparameters Tuning
The batch size can be set in `preprocess.py` **line 52**.

The learning rate of the two optimizers can be set in `train.py` **line 20 and 21**.

The weight of the adversarial loss can be adjusted in `train.py` **line 43** for training and **line 90** for testing.

