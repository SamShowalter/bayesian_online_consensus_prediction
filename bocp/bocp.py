#################################################################################
#
#             Project Title:  Bayesian Online Consensus Prediction
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import copy
import sys
import torch
import hashlib
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np
import itertools

from arguments import get_args, adjust_args, setup_ocp_args
from utils.summary import ResultStore
from utils.misc import (set_seed, setup_logger, print_args, get_configs,
                        make_list, update_args, BUDGET_ROSTER)
from utils.data import (
    read_pkl, write_pkl, cache_data, get_cache_filepath, zip_dict
)
from config.utils import make_config_combs
from selection_methods.roster import SELECTION_ROSTER
from prediction_methods.roster import PREDICTION_ROSTER
from datasets import (
    get_data, noise_experts, noise_models,
    DATASET_ROSTER,
)
from config import * # Imports all config arguments
logger = logging.getLogger(__name__)

#################################################################################
#   Function-Class Declaration
#################################################################################


class OnlineConsensusPredictionSimulation(object):

    def __init__(self, args):
        self.args = args
        self.seq_len = args.seq_len
        self.result_store = ResultStore(args)
        self.reset_data()

    def _get_data(self):

        # get_cache_filepath(args)
        if self.args.cache_data and os.path.exists(self.args.save_path):
            logger.info(f"Reading data file in from cache at: {self.args.save_path}")
            data = read_pkl(self.args.save_path)
        else: data = get_data(self.args)

        if self.args.noise_experts:
            noise_experts(self.args,data)
        if self.args.noise_models:
            noise_models(self.args, data)
        # cache_data(self.args,data)
        return data

    def reset_data(self):
        self.data=self._get_data()
        self.expert_preds = self.data['expert_preds']
        self.model_preds = self.data['model_preds']
        self.model_confs = self.data['model_confs']
        self.expert_confs =np.eye(self.args.num_classes)[self.expert_preds]
        self.model_preds = self.data['model_preds']
        self.num_models = self.model_preds.shape[0]
        self.num_experts = self.expert_preds.shape[0]
        self.num_feedback_sources = self.num_models + self.num_experts
        self.labels = self.data['targets']
        # Get sources
        self.sources = {
            "model": self.model_confs,
            "expert": self.expert_confs,
        }

        self.model_data = torch.zeros(
            (self.args.seq_len, self.args.num_classes)).to(self.args.device)
        self.expert_data = torch.zeros(
            (self.args.seq_len, self.args.num_classes)).to(self.args.device)

    def _get_data_indices(self):
        all_inds = np.arange(self.model_preds.shape[-1])
        # Don't shuffle for distribution shift
        if "distshift" in self.args.dataset:
            return all_inds[:self.seq_len]
        np.random.shuffle(all_inds)
        return all_inds[:self.seq_len]

    def check_if_complete(self):

        if not self.args.check_if_exists: return False
        save_path = os.path.join(
            self.args.out_root, self.args.experiment_name, "results.csv")
        return os.path.exists(save_path)

    # @njit(parallel=True)
    def simulate(self):

        if self.check_if_complete():
            logger.info(
                f"Experiment skip={self.args.check_if_exists} and found experiment, skipping")
            return

        # Select your selection and prediction methods
        self.sel_method = SELECTION_ROSTER[self.args.sel_method](self.args,self)
        self.pred_method = PREDICTION_ROSTER[self.args.pred_method](self.args,self)

        # iterate through the number of trials (simulations)
        for trial in tqdm(range(self.args.n_trials),
                          disable=self.args.disable_tqdm):

            # For case with accumulating cost
            self.args.total_cost = 0

            # For case with pre-set budget
            self.args.remaining_budget = self.args.budget
            self.args.curr_trial_id = trial

            # Simulate OAMS
            self.simulate_single_run(trial)

            # Save results
            if trial % self.args.save_iters == 0:
                self.result_store.save()

        self.result_store.save()

    def simulate_single_run(self,trial_id):
        # Shuffle data and truncate to sequence length
        self.reset_data()
        indices = self._get_data_indices()

        # Set timestep and reset selection / prediction methods
        self.args.timestep = 0
        self.args.total_cost = 0
        self.sel_method.reset()
        self.pred_method.reset()

        # Iterate through each timestep
        for i in tqdm(range(self.seq_len),
            disable=self.args.disable_tqdm):
            # print(self.args.total_cost)
            if (self.args.disable_tqdm and i %10 == 0):
                print(".",end="")

            # Simulate a single step
            self.simulate_step(trial_id,indices[i], i)
        print()

    def update_budget(self, feedback_type, amount=None):
        # Pre-set budget
        if self.args.preset_budget:
            self.args.remaining_budget -= (
                self.args.model_cost if feedback_type == "model"
                else self.args.expert_cost)

        # Cost just accumulates
        else:
            self.args.total_cost += (
                self.args.model_cost if feedback_type == "model"
                else self.args.expert_cost)

    def update_feedback(self, feedback_types, feedback_ids, feedback):

        if feedback_ids is not None: # May have had an uneven budget amount

            # Reconcile data formats to list
            feedback_types = make_list(feedback_types)
            feedback_ids = make_list(feedback_ids)

            # In case where all feedback types are same type
            if len(feedback_ids) > len(feedback_types):
                feedback_types *= len(feedback_ids)

            # Update budget and add feedback as list
            for feedback_type, feedback_id in zip(feedback_types, feedback_ids):
                # Updates budget ONE SAMPLE AT A TIME
                self.update_budget(feedback_type)

                # Dictionary of form
                # {
                # 'model': [<list of model ids>],
                # 'expert': [<list of expert ids>],
                # }
                self.feedback[feedback_type].append(feedback_id)

    def simulate_step(self, trial_id, sample, timestep):

        # Initial feedback and stopping condition
        stop = False
        self.sample_id = sample
        self.feedback = {"model": [], "expert":[]}
        self.args.timestep = timestep

        # Set budget of how much sampling is allowed
        self.sel_method.reset_sampling()

        # Until stopping condition is met, sample feedback
        while not stop:

            # Select both model and expert feedback
            stop, feedback_types, feedback_ids =\
                self.sel_method.select_feedback(sample)

            # Update the feedback based on the types and IDs
            self.update_feedback(feedback_types, feedback_ids, self.feedback)

        # Get labels for specific sample
        label = self.labels[sample]

        # Get prediction and prediction weights (and if we got GT)
        # From prediction method
        confidences, pred, weights, gt_queried = self.pred_method.predict(
            self.feedback,sample)

        # Update the selection method with the new feedback
        self.sel_method.update(sample, self.feedback, pred, label,weights, gt_queried)

        # Update the prediction method with the new feedback
        self.pred_method.update(sample, self.feedback, pred, label,weights, gt_queried)

        # This is for saving and reporting later
        if not self.args.preset_budget: remaining_budget = "n/a"
        else: remaining_budget = self.args.remaining_budget
        orig_confidences = self.model_confs[self.feedback['model'][0],sample].tolist()
        self.result_store.add_record(self.args,
            self.sel_method, self.pred_method,
            trial_id, timestep, sample, pred, label,
            confidences, int(pred==label), orig_confidences, weights,
            self.feedback['expert'], remaining_budget, self.args.total_cost, weights)

#######################################################################
# Main Method
#######################################################################

def main(logger,dataset, prior_method, prior_heur="err", inf_method=None):
    args = get_args()

    args.dataset = dataset
    args.prior_method = prior_method
    args.prior_heur = prior_heur

    base_args, config_combs = get_configs(globals(),args)
    args.prior_method = prior_method
    args.prior_heur = prior_heur
    if inf_method==None:
        args.inference_method = args.prior_method
    else: args.inference_method = inf_method

    config = None
    for config_comb in config_combs:
        config = zip_dict(base_args, config_comb)
        args = adjust_args(args,config)
        args.mhg_learn_bias=True
        args.check_if_exists = True
        args.n_trials = 1

        setup_ocp_args(args)
        logger = setup_logger(args)
        logger.info("====="*10)
        logger.info("Experiment Arguments")
        logger.info("====="*10)

        if args.seed is not None:
            set_seed(args)

        print_args(dict(args._get_kwargs()))
        # sys.exit(1)
        logger.info("====="*10)
        logger.info("Beginning Simulation for:")
        logger.info(f" - Experiment: {args.experiment_name}")
        logger.info(f" - Sel. Method: {args.sel_method}")
        logger.info(f" - Pred Method: {args.pred_method}")
        logger.info("====="*10)
        oms = OnlineConsensusPredictionSimulation(args)

        oms.simulate()

def main_single_run():
    args = get_args()

    setup_ocp_args(args)
    logger = setup_logger(args)
    logger.info("====="*10)
    logger.info("Experiment Arguments")
    logger.info("====="*10)

    if args.seed is not None:
        set_seed(args)

    print_args(dict(args._get_kwargs()))
    logger.info("====="*10)
    logger.info("Beginning Simulation for:")
    logger.info(f" - Experiment: {args.experiment_name}")
    logger.info(f" - Sel. Method: {args.sel_method}")
    logger.info(f" - Pred Method: {args.pred_method}")
    logger.info("====="*10)
    oms = OnlineConsensusPredictionSimulation(args)

    oms.simulate()


if __name__ == "__main__":
    main_single_run()

