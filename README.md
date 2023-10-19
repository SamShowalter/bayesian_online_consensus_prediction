# Bayesian Online Learning for Consensus Prediction
Code for AI Stats Submission "Bayesian Online Learning for Consensus Prediction"

## Setup

`pip install -r requirements.txt`

### Datasets
Our experiments make use of 2 large-scale crowdsourced datasets. Links for downloading these files can be found below. However, we also include pickle files of the model predictions and expert annotations for each dataset in this repository since they are relatively small. These files are all that are needed to run experiments.
- [ImageNet-16H](https://osf.io/2ntrf/)
- [CIFAR-10H](https://github.com/jcpeterson/cifar-10h)

### Models

We include the predictions of the trained models we utilized in our experiments. Those can be found within the datasets themselves in the `/data` directory

### Required Packages

Our code requires the following packages:
- [PyTorch](https://pytorch.org/)
- [Scipy](https://github.com/scipy/scipy)
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numba](https://numba.pydata.org/)
- argparse (CLI)
- yaml

### Other Details 

Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.8. 

## Running Experiments

All of our experiments and ablations were completed through a single argparse CLI. There are two ways to conduct experiments in this setting. The first is to run a custom script. An example of this for ImageNet-16H is shown below.

```bash
python bocp/bocp.py \
  --num-models 1 \
  --num-experts 6 \
  --dataset "imagenet16h" \
  --sel-method "mhg" \      # Options: mhg, random, entropy, mp
  --pred-method "mhg" \     # Options: mhg, even_weight, ftl, mp (lined up to correspond with above)
  --prior-method "finset" \ # Finite Experts setting (options: finset, infset, fixed)
  --prior-heur "err" \      # Error rate posterior heuristic
```

The second, and simpler alternative is run one of the pre-made scripts we have placed in the `scripts` directory. These scripts will re-create our experimental results and are named in a composable manner. Please utilize the following legend to identify the script your are interested in running
- `c10h` - CIFAR-10H
- `i16h` - ImageNet-16H
- `baselines` - File to run all of the included baselines
- `distshift` - Distribution shift
- `fs` - FinSet 
- `is` - InfSet
- `fixed` - Fixed prior (corresponding `fs` and `is` represent the inference procedure for these scripts)


