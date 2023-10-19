#################################################################################
#
#             Project Title:  Arguments
#             Date:           2023.05.22
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import numpy as np
import ast
import argparse

from utils.misc import BUDGET_ROSTER, compute_mhg_param
from utils.data import get_cache_filepath, zip_args

#######################################################################
# Helper functions
#######################################################################

def _str2arr(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isintance(arr, list):
        return np.array(list)
    if isinstance(arr, str):
        arr = ast.literal_eval(arr)
        return np.array(arr)
    assert False, \
        f"Input {arr} not an array type"


#################################################################################
#   Main Method
#################################################################################


def get_data_args(parser):

    parser.add_argument('--num-classes', default=16, type=int,
                        help='Number of classes')
    parser.add_argument('--total-num-samples', default=10000, type=int,
                        help='Total number of samples if generated synthetically')
    parser.add_argument('--true-target-dist', default=None, type=_str2arr,
                        help='Total number of samples if generated synthetically')
    parser.add_argument('--dataset', default='imagenet16h', type=str,
                        help='Type of dataset')
    parser.add_argument('--data-path', default=None, type=str,
                        help='Path to dataset if relevant')
    parser.add_argument('--expert-data', default=None, type=str,
                        help='Data source for expertsread_pkl(self.expert_preds_path)')
    parser.add_argument('--model-data', default=None, type=str,
                        help='Whether or not model data is synthetic')
    parser.add_argument('--target_model_perf', default=None, type=_str2arr,
                        help='Synthetic model distribution or noised distribution')
    parser.add_argument('--target-expert_perf', default=None, type=_str2arr,
                        help='Synthetic expert target performance (or from noising)')
    parser.add_argument('--imagenet-clean-experts', action="store_true", default=False,
                        help='Whether or not experts should be clean')
    parser.add_argument('--noise-models', action="store_true", default=False,
                        help='Whether or not to noise the input model predictions')
    parser.add_argument('--noise-experts', action="store_true", default=False,
                        help='Whether or not to noise the input expert predictions')
    parser.add_argument('--cache-data', action="store_true", default=False,
                        help='Whether or not to save data from noising or generation')
    parser.add_argument('--cache-root', default = os.path.abspath(os.path.join(__file__,"../../data/cache")),
                        help='Whether or not to save data from noising or generation')
    parser.add_argument('--save-iters', default=5,
                        type = int,
                        help='After how many iterations to save data')
    parser.add_argument('--check-if-exists',action="store_true",
                        default=False,
                        help='Should we check if the file exists instead of re-running?')

def get_model_args(parser):

    parser.add_argument('--model-id-sel-method', default="random",
                        help='model id selection method')
    parser.add_argument('--des-model-perfs', default=None,
                        type=list,nargs="+",
                        help='Desired model perfs (list) of size num_models')
    parser.add_argument('--acc-inc-forecast-window', default=3,
                        type = int,
                        help='Error reduction forecast window')
    parser.add_argument('--prior-method', default="finset",
                        choices=['finset','infset','baselines', 'fixed'],
                        help='Prior formulation method')
    parser.add_argument('--fixed-a-prior', default=1.0,
                        type=float,
                        help='Default weight for model')
    parser.add_argument('--prior-heur', default="err",
                        choices=['err','err_red'],
                        help='Heuristic for proposed method')
    parser.add_argument('--posterior-acc-inc', default=0.04,
                        type=float,
                        help='Posterior error reduction fraction')
    parser.add_argument('--posterior-error-rate', default=0.1,
                        type=float,
                        help='Error rate of posterior for mode')
    parser.add_argument('--prior-lr', default=1e-3,
                        type=float,
                        help='Learning rate for identifying prior parameters')

    parser.add_argument('--prior-normal-mu', default=0.1,
                        type=float,
                        help='Normal Mu parameters')
    parser.add_argument('--prior-normal-sigma', default=1,
                        type=float,
                        help='Normal sigma parameters')
    parser.add_argument('--prior-dist', default='mixed',
                        choices = ['gamma','normal'],
                        help='Normal sigma parameters')
    parser.add_argument('--prior-gamma-conc', default=2,
                        type=float,
                        help='gamma concentration parameters')
    parser.add_argument('--prior-gamma-rate', default=1,
                        type=float,
                        help='prior-gamma-rate parameter value')
    parser.add_argument('--a0-prior-gamma-conc', default=3,
                        type=float,
                        help='gamma concentration parameters')
    parser.add_argument('--a0-prior-gamma-rate', default=2,
                        type=float,
                        help='prior-gamma-rate parameter value')
    parser.add_argument('--prior-lognormal-mu', default=0,
                        type=float,
                        help='Lognormal Mu parameters')
    parser.add_argument('--prior-lognormal-sigma-sq', default=1,
                        type=float,
                        help='prior-lognormal-sigma-sq parameter value')
    parser.add_argument('--err-red-num-mc-samples', default=100,
                        type=float,
                        help='Number of MC samples for error reduction prediction')
    parser.add_argument('--prior-num-mc-samples', default=100,
                        type=float,
                        help='Number of MC samples from prior')
    parser.add_argument('--mhg-num-mc-samples', default=1000,
                        type=float,
                        help='Number of MC samples to determine exp. error rate')
    parser.add_argument('--prior-convergence-check-iters', default=20,
                        type=int,
                        help='Iterations before checking again for convergence')
    parser.add_argument('--prior-convergence-tol', default=1e-5,
                        type=float,
                        help='Convergence tolerance to stop training')
    parser.add_argument('--prior-recompute-iters', default=1,
                        type=float,
                        help='Number of iterations to recompute prior')
    parser.add_argument('--prior-param-mapping-func', default="softplus",
                        help  = "Mapping function to make sure params are non-negative")
    parser.add_argument('--prior-max-training-iters', default=1000,
                        type=int,
                        help='Maximum training iterations for prior')
    parser.add_argument('--prior-data-window-size', default=500,
                        type=int,
                        help='size of the data window for computing the prior')
    parser.add_argument('--prior-warmup-window', default=0,
                        type=int,
                        help='Size of warmup window')
    parser.add_argument('--model-retrain-sleep-window', default=50,
                        type=int,
                        help='Logistic regression sleep window')
    parser.add_argument('--model-timestep-start', default=100,
                        type=int,
                        help='After how many timesteps to utilize model')
    parser.add_argument('--chosen-model-id', default=0,
                        type=int,
                        help='Chosen model ID')

def get_ocp_args(parser):
    parser.add_argument('--seq-len', default=1000, type=int,
                        help='Sequence length for simulation')
    parser.add_argument('--n-trials', default=1, type=int,
                        help='Number of trials per method')
    parser.add_argument('--sel-method', type=str, default='random',
                        help='Selection method for Online Consensus Prediction (OCP)')
    parser.add_argument('--pred-method', type=str, default='even_weight',
                        help='Prediction method for Online Consensus Preidiction (OCP)')
    parser.add_argument('--preset-budget',action="store_true", default=False,
                        help="Should the budget be pre-set, or should it just accumulate")
    parser.add_argument('--budget-func',default="percent_of_full",
                        help="AL budget as a function of the length of the sequence")
    parser.add_argument('--model-cost',default=0,
                        help="AL budget as a function of the length of the sequence")
    parser.add_argument('--expert-cost',default=1,
                        help="AL budget as a function of the length of the sequence")
    parser.add_argument('--budget-percent-of-full',default=0.90,
                        help="Percent of total annotation budget to use")
    parser.add_argument('--num-experts',default=3,
                        help="Number of (human) experts")
    parser.add_argument('--num-models',default=1,
                        help="Number of models")
    parser.add_argument('--num-model-queries',default=None,
                        help="Number of model queries (either None or set to max)")
    parser.add_argument('--num-expert-queries',default=None,
                        help="Number of expert queries needed in case where preset budget is not available. Used by random method only.")
    parser.add_argument('--expert-weight',default=1,
                        help="Weight of expert feedback by default")
    parser.add_argument('--full-feedback',default=1,
                        help="For bandit methods that need feedback, should they take all of it or not")
    parser.add_argument('--model-weight',default=1,
                        help="Weight of model by default")
    parser.add_argument('--rand-expert-query-perc',default = 0, type=int,
                        help="In unfixed budgeting, how many queries should random method take?")
    parser.add_argument('--model-query-perc',default = 1.0, type=float,
                        help="In unfixed budgeting, how many queries should random method take?")
    parser.add_argument('--best-pred-expert-only',action="store_true",
                        help="In best prediction, consider only experts")
    parser.add_argument('--best-pred-model-only',action="store_true",
                        help="In best prediction, consider only models")
    parser.add_argument('--identifiable-experts',action="store_false", # Default is True
                        help="Can you identify the specific expert you wish to query?")
    parser.add_argument('--single-expert',action="store_false", # Default is True
            help="Should just query a single expert instead of all of them for bandits. All are used otherwise.")
    parser.add_argument('--iterative-expert-gt',action="store_false", # Default is True
            help="Should expert gt be queried repeatedly (and randomly) until GT achieved?")
    parser.add_argument('--entropy-tuning-param',default=1.0, type=float,
                        help="Entropy tuning parameter")
    parser.add_argument('--mp-tuning-param',default=1.0, type=float,
                        help="MP entropy tuning parameter")
    parser.add_argument('--hyperparams',default={},
                        help="Logistic Regression Hyperparameters")
    parser.add_argument('--even-weight-all-feedback',action="store_true", # Default false
                        help="Whether or not to use both model and expert feedback together for even weight prediction (as opposed to one of them individually)")
    parser.add_argument('--curr-timestep',default=0, type=int,
                        help="Current timestep")
    parser.add_argument('--class-specific-belief',action="store_true",default=False,
                        help="Should the selection parameters be class-specific?")
    parser.add_argument('--mhg-learn-bias',action="store_false",
                        help="Learn the MHG bias terms")
    parser.add_argument('--mhg-hybrid-prior-method',default="finset",
                        help="Hybrid prior method")
    parser.add_argument('--mhg-hybrid-prior-heur-method',default="infset",
                        help="Hybrid prior method for heuristic")
    parser.add_argument('--inference-method',default=None,
                        choices=[None, "finset","infset"],
                        help="Inference method")


def get_gen_args(parser):
    parser.add_argument('--cpu', action="store_false",
                        help='Are we using just the CPU?')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU device')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of CPU workers')
    parser.add_argument('--disable-tqdm', default=0,type=int,
                        help="Disable tqdm or not")
    parser.add_argument('--seed', default=5, type=int,
                        help="random seed")
    parser.add_argument('--out-root',
                        default=os.path.abspath(os.path.join(__file__,"../../results")),
                        help='directory to output the results')

def get_budget(args):
    if not args.preset_budget:
        args.budget = float("inf") # Budget is not capped, can sample forever
        assert args.iterative_expert_gt or args.num_model_queries is not None,\
            ("Number of model queries is None but budget is not preset. "+
             "Either pre-set the budget or set a number of model queries")
        assert args.iterative_expert_gt or args.num_expert_queries is not None,\
        ("Number of expert queries is None but budget is not preset and iterative gt not used. "+
        "Either pre-set the budget or set a number of expert queries or set iterative gt")
    else:
        args.budget=  BUDGET_ROSTER[args.budget_func](args)
    return args.budget

def get_model_id_sel_method(args):
    if args.model_id_sel_method == 'random':
        return args.model_id_sel_method
    elif args.model_id_sel_method == "id":
        return f"id_{args.chosen_model_id}"
    elif args.model_id_sel_method == "perf":
        perf_avg = np.sum(args.des_model_perfs)/args.num_models
        return f"perf{perf_avg:.2f}"

def setup_ocp_args(args):
    if os.path.exists(args.dataset):
        dataset = args.dataset.split("/")[-1].split(".")[0]
    else: dataset = args.dataset

    args.experiment_name = (
         f"{dataset.replace('-','_')}-{args.sel_method}-{args.pred_method}-" +
         f"m{args.num_models}-e{args.num_experts}-" +
         f"mc{args.model_cost}-ec{args.expert_cost}-l{args.seq_len}-" +
         f"cs{int(args.class_specific_belief)}-" +
         f"m.id_sel.{get_model_id_sel_method(args)}-" +
         f"s.rand.m{args.model_query_perc}-" +
         f"s.rand.e{args.rand_expert_query_perc}-" +
         f"s.entropy{args.entropy_tuning_param}-" +
         f"s.mp{args.mp_tuning_param}-" +
         f"s.mhg.prior.{(args.prior_method.replace('baselines','') + '_' + args.inference_method if args.inference_method is not None else '')}-" +
         f"s.mhg.h{args.prior_heur}-" +
         f"s.mhg.p{compute_mhg_param(args)}-" +
         f"igt{int(args.iterative_expert_gt)}-rs{args.seed}-" +
         f"warm{args.prior_warmup_window}-" +
         f"g.p{args.prior_gamma_conc}{args.prior_gamma_rate}-" +
         f"b{int(args.mhg_learn_bias)}")
    args.budget = get_budget(args)
    args.budget_per_sample = (args.budget/args.seq_len)
    args.remaining_budget = args.budget

def get_args():

    parser = argparse.ArgumentParser(
        description='Online Active Model Selection')

    get_gen_args(parser)
    get_ocp_args(parser)
    get_model_args(parser)
    get_data_args(parser)

    args = parser.parse_args()
    return args

def adjust_args(args, config=None):
    if config: zip_args(args,config)
    setup_ocp_args(args)
    get_cache_filepath(args)
    return args

