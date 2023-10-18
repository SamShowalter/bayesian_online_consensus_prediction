#################################################################################
#
#             Project Title:  Configurations for Imagenet-16h
#             Date:           2023-09-26
#
#################################################################################

import numpy as np

#######################################################################
# General Parameters
#######################################################################

# 45
dynamic_args = {
    "seq_len" : [4800],
    "num_experts" : [3, 6],
    "num_models": [1], #[1],
    "des_model_perfs" : [[k] for k in [  0.5, 0.7, 0.9]],
    # "des_model_perfs" : [[k] for k in [  0.45, 0.65,0.85]],
}


# 45 * 6 * 5 = 45*30 = 450*3 = 1350
sel_pred_methods = [
        ["random","even_weight"],
        ["qbc","ftl"],
        ["mp","mp"],
    ]

sel_method_param_lkp = {
    "mp": "mp_tuning_param",
    "qbc": "qbc_entropy_tuning_param",
    "random": "rand_expert_query_perc", # Need to add this parameter to random
}

model_hyperparams = {
    "rand_expert_query_perc": list(np.arange(0.005,1,0.01)),
    "qbc_entropy_tuning_param": list(np.arange(0,10,0.2)) + [ 50, 100,500, 1000, 10000, 50000, 100000, 1e10],
    "mp_tuning_param": list(np.arange(0,10,0.2))+ [10, 50, 100,500, 1000, 10000, 50000, 100000, 1e10],
}

#################################################################################
# Single Model Configs
#################################################################################

DEVICE=2
base_args = {

    "check_if_exists": True,

    "dataset" : "imagenet16h",
    "disable_tqdm": False,
    "num_classes": 16,
    "synthetic_experts" : False,
    "synthetic_models" : False,
    "seed":5,
    "noise_experts" : False,
    "noise_models" : False,
    "model_query_perc": 1.0,
    "n_trials" : 10,
    "device": f"cuda:{DEVICE}",
    "model_cost": 0,
    "expert_cost": 1,
    "preset_budget": False,
    "budget_percent_of_full": 0,
    "rand_model_query_perc": 1,
    "class_specific_belief": False,
    "model_id_sel_method": "perf",

}

