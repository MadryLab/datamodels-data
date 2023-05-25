# Data from "Datamodels: Predicting Predictions with Training Data"

Here we provide the data used in the paper "Datamodels: Predicting Predictions with Training Data" ([arXiv](https://arxiv.org/abs/2202.00622), [Blog](https://gradientscience.org/datamodels-1)).

Looking for the code to make your own datamodels? It's now been released [here](https://github.com/MadryLab/datamodels)!

*Note that all of the data below is stored on Amazon S3  using the “requester pays” option to avoid a blowup in our data transfer costs (we put estimated AWS costs below)---if you are on a budget and do not mind waiting a bit longer, please contact us at datamodels@mit.edu and we can try to arrange a free (but slower) transfer.*

## Citation
To cite this data, please use the following BibTeX entry:
```
@inproceedings{ilyas2022datamodels,
  title = {Datamodels: Predicting Predictions from Training Data},
  author = {Andrew Ilyas and Sung Min Park and Logan Engstrom and Guillaume Leclerc and Aleksander Madry},
  booktitle = {ArXiv preprint arXiv:2202.00622},
  year = {2022}
}
```

## Overview
We provide the data used in our paper to analyze two image classification datasets: CIFAR-10 and (a modified version of) [FMoW](https://wilds.stanford.edu/datasets/#fmow).

For each dataset, the data consists of two parts:
1. *Training data* for datamodeling, which consists of:
     * Training subsets or "training masks", which are the independent variables of the regression tasks; and
     * Model outputs (correct-class margins and logits), which are the
dependent variables of the regression tasks.
2. *Datamodels* estimated from this data using LASSO.

For each dataset, there are multiple versions of the data depending on the choice of the hyperparameter &alpha;, the subsampling fraction (this is the random fraction of training examples on which each model is trained; see Section 2 of our paper for more information).

Following table shows the number of models we trained and used for estimating datamodels (also see Table 1 in paper):
| Subsampling &alpha; (%) | CIFAR-10  | FMoW    |
|-----------------------|-----------|---------|
| 10                   | 1,500,000 | N/A     |
| 20                   | 750,000   | 375,000 |
| 50                   | 300,000   | 150,000 |
| 75                  | 600,000   | 300,000 |


### Training data
For each dataset and $\alpha$, we provide the following data:

```python
# M is the number of models trained
/{DATASET}/data/train_masks_{PCT}pct.npy  # [M x N_train] boolean
/{DATASET}/data/test_margins_{PCT}pct.npy # [M x N_test] np.float16
/{DATASET}/data/train_margins_{PCT}pct.npy # [M x N_train] np.float16
```
(The files live in the Amazon S3 bucket `madrylab-datamodels`; we provide instructions for acces in the <a href="#downloading">next section</a>.)

Each row of the above matrices corresponds to one instance of model trained; each column corresponds to a training or test example.
CIFAR-10 examples are organized in the default order; for FMoW, see <a href="#fmow-data">here</a>.
For example, a train mask for CIFAR-10 has the shape [M x 50,000].

For CIFAR-10, we also provide the full logits for all ten classes:
```python
/cifar/data/train_logits_{PCT}pct.npy  # [M x N_test x 10] np.float16
/cifar/data/test_logits_{PCT}pct.npy   # [M x N_test x 10] np.float16
```
Note that you can also compute the margins from these logits.

We include an addtional 10,000 models for each setting that we used for evaluation; the total number of models in each matrix is `M` as indicated in the above table plus 10,000.

### Datamodels
All estimated datamodels for each split (`train` or `test`) are provided as a dictionary in a `.pt` file (load with `torch.load`):
```python
/{DATASET}/datamodels/train_{PCT}pct.pt
/{DATASET}/datamodels/test_{PCT}pct.pt
```

Each dictionary contains:
* `weight`: matrix of shape `N_train x N`, where `N` is either `N_train` or `N_test` depending on the group of target examples
* `bias`: vector of length `N`, corresponding to biases for each datamodel
* `lam`: vector of length `N`, regularization &lambda; chosen by CV for each datamodel

## Downloading
We make all of our data available via Amazon S3.
Total sizes of the training data files are as follows:
| Dataset, 	&alpha; (%) | masks, margins (GB) |  logits (GB) |
|-----------------------|-----------|---------|
| CIFAR-10, 10           | 245 | 1688 |
| CIFAR-10, 20           | 123 | 849 |
| CIFAR-10, 50           | 49 | 346 |
| CIFAR-10, 75           | 98 | 682 |
| FMoW, 20           | 25.4 | -  |
| FMoW, 50           | 10.6 | -  |
| FMoW, 75           | 21.2 | -  |

Total sizes of datamodels data (the model weights) are 16.9 GB for CIFAR-10 and 0.75 GB for FMoW.

### Setting up AWS
1. Make an AWS account
2. Download the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. Run `aws configure` and add the Access key ID (you can get these by clicking on your account on top right corner -> Security credentials)


### API
You can download them using the Amazon S3 CLI interface with the requester pays option as follows (replacing the fields {...} as appropriate):
```bash
aws s3api get-object --bucket madrylab-datamodels \
                     --key {DATASET}/data/{SPLIT}_{DATA_TYPE}_{PCT}.npy \
                     --request-payer requester \
                     [OUT_FILE]
```

For example, to retrieve the test set margins for CIFAR-10 models trained on 50% subsets, use:
```bash
aws s3api get-object --bucket madrylab-datamodels \
                     --key cifar/data/test_margins_50pct.npy \
                     --request-payer requester \
                     test_margins_50pct.npy
```

### Pricing
The total data transfer fee (from AWS to internet) for all of the data is around $374 (= 4155 GB x 0.09 USD per GB).

If you only download everything except for the logits (which is sufficient to reproduce all of our analysis), the fee is around $53.

## Loading data

The data matrices are in `numpy` array format (`.npy`).
As some of these are quite large, you can read small segments without reading the entire file into memory
by additionally specifying the `mmap_mode` argument in `np.load`:
```python
X = np.load('train_masks_10pct.npy', mmap_mode='r')
Y = np.load('test_margins_10pct.npy', mmap_mode='r')
...
# Use segments, e.g, X[:100], as appropriate
# Run regress(X, Y[:]) using choice of estimation algorithm.
```

## FMoW data

We use a customized version of the FMoW dataset from [WILDS](https://wilds.stanford.edu/datasets/#fmow) (derived from this [original dataset](https://arxiv.org/abs/1711.07846)) that restricts the year of the training set to 2012. Our code is adapted from [here](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/fmow_dataset.py).

To use the dataset, first download WILDS using:
```bash
pip install wilds
```
(see [here](https://github.com/p-lambda/wilds#installation) for more detailed instructions).

In our paper, we only use the in-distribution training and test splits in our analysis (the original version from WILDS also has out-of-distribution as well as validation splits).
Our dataset splits can be constructed as follows and used like a PyTorch dataset:
```python
from fmow import FMoWDataset

ds = FMoWDataset(root_dir='/mnt/nfs/datasets/wilds/',
                     split_scheme='time_after_2016')

transform_steps = [
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform = transforms.Compose(transform_steps)

ds_train = ds.get_subset('train', transform=transform)
ds_test = ds.get_subset('id_test', transform=transform)
```

The columns of matrix data <a href="#training-data">described above</a> is ordered according to the default ordering of examples given by the above constructors.
