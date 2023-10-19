#################################################################################
#
#             Project Title:  Configurations for CIFAR10-H
#             Date:           2023-09-26
#
#################################################################################

import numpy as np

#######################################################################
# General Parameters
#######################################################################

dynamic_args = {
    "seq_len" : [10000],
    "num_experts" : [3, 10, 50],
    "num_models": [1],
    "des_model_perfs" : [[k] for k in [0.5, 0.7,0.9]],
}


sel_pred_methods = [
        ["entropy","ftl"],
        ["random","even_weight"],
        ["mp","mp"],
    ]

sel_method_param_lkp = {
    "mp": "mp_tuning_param",
    "entropy": "entropy_tuning_param",
    "random": "rand_expert_query_perc",
}

model_hyperparams = {
    "rand_expert_query_perc": list(np.arange(0,1,0.05)),
    "entropy_tuning_param": list(np.arange(0,10,0.5)),
    "mp_tuning_param":list(np.arange(0,10,0.5)),
}

#################################################################################
# Single Model Configs
#################################################################################

DEVICE=0
base_args = {

    "expert_data" : "cifar10h",
    "disable_tqdm": False,
    "num_classes": 10,
    "seed":5,
    "model_query_perc": 1.0,
    "n_trials" : 10,
    "device": f"cuda:{DEVICE}",
    "model_cost": 0,
    "expert_cost": 1,
    "rand_model_query_perc": 1,
    "model_id_sel_method": "perf",
}

