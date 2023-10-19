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
    "seq_len" : [2400],
    "num_experts" : [6],
    "num_models": [1],
    "chosen_model_id" : list(range(5)),
}


sel_pred_methods = [
        ["entropy","ftl"],
        ["random","even_weight"],
        ["mp","mp"],
        ["mhg","mhg"],
    ]

sel_method_param_lkp = {
    "mp": "mp_tuning_param",
    "entropy": "entropy_tuning_param",
    "random": "rand_expert_query_perc",
    "mhg": "posterior_error_rate",
}

model_hyperparams = {
    "posterior_error_rate":list(np.arange(0.0, 1,0.05)),
    "rand_expert_query_perc": list(np.arange(0,1,0.05)),
    "entropy_tuning_param": list(np.arange(0,10,0.5)),
    "mp_tuning_param": list(np.arange(0,10,0.5))
}

#################################################################################
# Single Model Configs
#################################################################################

DEVICE=6
base_args = {

    "check_if_exists": True,

    "dataset" : "imagenet16h_distshift",
    "disable_tqdm": False,
    "num_classes": 16,
    "seed":5,
    "model_query_perc": 1.0,
    "n_trials" : 1,
    "device": f"cuda:{DEVICE}",
    "model_cost": 0,
    "expert_cost": 1,
    "rand_model_query_perc": 1,
    "class_specific_belief": False,
    "model_id_sel_method": "id",

   # Parameters for proposed method
    "prior_gamma_conc" : 3,
    "prior_gamma_rate" : 2,
    "a0_prior_gamma_conc" : 3,
    "a0_prior_gamma_rate" : 2,
    "prior_param_mapping_func": "softplus",
    "prior_convergence_check_iters": 20,
    "prior_recompute_iters": 20,
    "prior_data_window_size": 100,
    "prior_num_mc_samples": 100,
    "prior_lr" : 1e-2,
    "num_mc_samples" :1000,
    "prior_convergence_tol" : 0.01,
    "prior_method": "infset",
}


