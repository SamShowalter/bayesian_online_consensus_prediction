'''
Some helper functions for Experiments
'''
import logging
import random
import os
import math
import shutil
import torch
import numpy as np
import itertools

logger = logging.getLogger(__name__)

#######################################################################
# Budget functions
#######################################################################


def make_config_combs(dynamic_dict, sel_pred_methods=None,
                      model_params = None, model_param_lkp=None):
    permutation_dicts = []
    if sel_pred_methods:
        for sel_method, pred_method in sel_pred_methods:
            dynamic_dict["sel_method"] = [sel_method]
            dynamic_dict["pred_method"] = [pred_method]
            sel_method_param_key = model_param_lkp[sel_method]
            dynamic_dict[sel_method_param_key] = model_params[
                sel_method_param_key]
            permutation_dicts += _make_config_combs(dynamic_dict)
            del dynamic_dict[sel_method_param_key]

    else: permutation_dicts += _make_config_combs(dynamic_dict)
    return permutation_dicts

def _make_config_combs(dynamic_dict):
    keys, values = zip(*dynamic_dict.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutation_dicts


def get_configs(glbals, args):

    # ONly two datasets, stitch fix
    dataset = args.dataset
    if "imagenet16h" not in args.dataset:
        dataset = "cifar10h"

    module_lkp = {
        "imagenet16h":"i16h",
        "imagenet16h_distshift":"i16h",
        "imagenet16h_clean":"i16h",
        "cifar10h": "c10h",
        "finset": "fs",
        "infset": "is",
        "fixed": "fs",
        "hybrid": "hybrid",
        }
    dset = module_lkp[dataset]
    if args.prior_method == "baselines": module_name = f"{dset}_baselines"
    elif args.prior_method == "distshift": module_name = f"{dset}_distshift"
    else:
        meth = module_lkp[args.prior_method]
        heur = args.prior_heur.replace("_flat","") # Flat has same config
        module_name = f"{dset}_{meth}_{heur}"
    args.config_file = module_name
    config_combs = make_config_combs(
        getattr(glbals[module_name],"dynamic_args"),
        getattr(glbals[module_name],"sel_pred_methods"),
        getattr(glbals[module_name],"model_hyperparams"),
        getattr(glbals[module_name],"sel_method_param_lkp"),
    )
    return getattr(glbals[module_name], "base_args"), config_combs

def make_list(item):
    if isinstance(item,np.ndarray): item=item.tolist()
    if not isinstance(item, list): item = [item]
    return item

def budget_simple_multiple(args):
    return args.seq_len * args.budget_mult

def budget_percent_of_full(args):
    full_price_per_sample = args.expert_cost*args.num_experts + args.model_cost*args.num_models
    full_price = full_price_per_sample*args.seq_len

    return args.budget_percent_of_full*full_price

BUDGET_ROSTER = {
    "percent_of_full": budget_percent_of_full,
    "multiple": budget_simple_multiple,
}


#######################################################################
# Other helper functions
#######################################################################

def compute_mhg_param(args):
    param = {
        "err":args.posterior_error_rate,
        "err_red":args.posterior_acc_inc,
    }
    return param[args.prior_heur]

def update_args(args,arg_dict):
    for k, v in arg_dict.items():
        if k not in args.__dict__:
            assert False, f"Key {k} not found in Args"
        args.__dict__[k] = v

def print_args(args):
    max_arg_len = max(len(k) for k, v in args.items())
    key_set = sorted([k for k in args.keys()])
    for k in key_set:
        v = args[k]
        logger.info("{} {} {}".format(
            k,
            "." * (max_arg_len + 3 - len(k)),
            v,
        ))
def setup_logger(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Exchange with print args
    # print_args(dict(args._get_kwargs()))

    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)

def randargmax(b,axis=-1):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape[axis])
                     *(b == np.amax(b,axis=axis, keepdims=True)),axis=axis)

def torch_rand_argmax(x,axis):

    max_values = x.max(dim=axis).values
    y = torch.zeros_like(x).float().to(x.device)
    z = torch.rand(*x.shape).to(x.device)

    mask = (x == max_values.unsqueeze(-1))
    y[mask] = z[mask]
    return torch.argmax(y,axis=axis)


# if __name__ == "__main__":
#     a = np.random.randint(0,3,size=(4))
#     print(a)
#     print(randargmax(a, axis=0))

