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
    "num_models": [1],
    "des_model_perfs" : [[k] for k in [  0.5, 0.7, 0.9]],
}


# 45 * 6 * 5 = 45*30 = 450*3 = 1350
sel_pred_methods = [
        ["mhg","mhg"],
    ]

sel_method_param_lkp = {
    "mhg": "posterior_error_rate",
}

model_hyperparams = {
    "posterior_error_rate": list(np.arange(0,1.0,0.05)),
}

#################################################################################
# Single Model Configs
#################################################################################

DEVICE=7
base_args = {

    "dataset" : "imagenet16h",
    "disable_tqdm": False,
    "num_classes": 16,
    "seed":5,
    "model_query_perc": 1.0,
    "n_trials" : 1,
    "device": f"cuda:{DEVICE}",
    "model_cost": 0,
    "expert_cost": 1,
    "rand_model_query_perc": 1,
    "model_id_sel_method": "perf",

   # Parameters for proposed method
    "prior_gamma_conc" : 3,
    "prior_gamma_rate" : 2,
    "prior_param_mapping_func": "softplus",
    "prior_convergence_check_iters": 100,
    "prior_lr" : 0.1,
    "prior_recompute_iters": 20,
    "prior_data_window_size": 100,
    "prior_num_mc_samples": 100,
    "num_mc_samples" :1000,
    "prior_convergence_tol" : 0.01,
}


