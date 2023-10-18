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

# 45
dynamic_args = {
    "seq_len" : [10000],
    # "num_experts" : [10,50],
    "num_experts" : [10,50],
    "num_models": [1], #[1],
    "des_model_perfs" : [[k] for k in [0.5, 0.7,0.9]],
}


# 45 * 6 * 5 = 45*30 = 450*3 = 1350
sel_pred_methods = [
        ["mhg","mhg"],
    ]

sel_method_param_lkp = {
    "mhg": "posterior_acc_inc",
}

model_hyperparams = {
    "posterior_acc_inc": list(reversed(list(np.arange(0,0.10,0.1))))
    # "posterior_acc_inc": list(np.arange(0.005,0.03,0.003)) + [0.08, 0.15],
    # "posterior_acc_inc": list(reverse(list(np.arange(0.015,0.04,0.003)),
    # "posterior_acc_inc": list(np.arange(0,0.03,0.003))
}

#################################################################################
# Single Model Configs
#################################################################################

DEVICE=7
base_args = {
    "check_if_exists": True,
    "dataset" : "/home/showalte/research/oams/data/cifar10/cifar10-fixmatch-rs1-s40.pkl",
    "expert_data" : "cifar10h",
    "disable_tqdm": False,
    "num_classes": 10,
    "synthetic_experts" : False,
    "synthetic_models" : False,
    "seed":5,
    "noise_experts" : False,
    "noise_models" : False,
    "model_query_perc": 1.0,
    "n_trials" : 1,
    "device": f"cuda:{DEVICE}",
    "model_cost": 0,
    "expert_cost": 1,
    "preset_budget": False,
    "budget_percent_of_full": 0,
    "rand_model_query_perc": 1,
    "class_specific_belief": False,
    "model_id_sel_method": "perf",

   # Parameters for proposed method
    "prior_gamma_conc" : 3,
    "prior_gamma_rate" : 2,
    "prior_param_mapping_func": "softplus",
    "prior_convergence_check_iters": 100,
    "prior_lr" : 0.1,
    "prior_recompute_iters": 20,
    "prior_num_mc_samples": 100,
    "prior_data_window_size": 100,
    "err_red_num_mc_samples": 100,
    "num_mc_samples" :1000,
    "prior_convergence_tol" : 0.01,
}
