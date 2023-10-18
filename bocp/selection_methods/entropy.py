#################################################################################
#
#             Project Title:  Entropy Selector
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import sys
import numpy as np
import scipy.stats as stats
import logging

from . import BaseBOCPSelector
from .utils import entropy_mp_update, entropy_select_model_feedback, entropy_mp_select_expert_feedback

logger = logging.getLogger(__name__)

#######################################################################
# Query By Committee selector
#######################################################################

class EntropySelector(BaseBOCPSelector):

    def __init__(self,args,simulator, init_weight=0):
        BaseBOCPSelector.__init__(self,args,simulator)
        self.name = "entropy"
        self.init_weight = init_weight
        self.reset()


    def _make_model_expert_belief(self):
        if not self.args.class_specific_belief:
            self.model_belief = self.init_weight*np.ones(self.args.num_models)
            self.expert_belief = self.init_weight*np.ones(self.args.num_experts)
        else:
            self.model_belief = self.init_weight*np.ones(
                (self.args.num_models, self.args.num_classes))
            self.expert_belief = self.init_weight*np.ones(
                (self.args.num_experts, self.args.num_classes))

    def reset_sampling(self):
        self.num_model_queries = (
            np.random.rand(self.args.num_models) <=
            self.args.model_query_perc).astype(int).sum()
        self.sample_expert_flag=False
        self.models_already_seen = set()
        self.experts_already_seen = set()
        self.query_expert = None

    def reset(self):

        self._make_model_expert_belief()
        self.reset_sampling()

    def update(self, sample_id,
        feedback, pred, label, weights, gt_queried,
        *args, **kwargs):
        entropy_mp_update(sample_id,
            feedback, pred, label, weights, gt_queried,
            self.simulator.expert_confs, self.simulator.model_confs,
            self.expert_belief, self.model_belief,
            self.args.num_experts, self.args.num_models,
            class_specific_belief=self.args.class_specific_belief,
        )

    def _select_model_feedback(self, num_queries=1, eps=1e-5, *args, **kwargs):

        choices, self.num_model_queries = entropy_select_model_feedback(
            num_queries, eps,
            self.expert_belief,self.model_belief,
            self.args.num_experts, self.args.num_models,
            self.models_already_seen, self.num_model_queries,
            self.args.remaining_budget, self.args.model_cost,
            class_specific_belief=self.args.class_specific_belief,
        )
        return choices

    def _select_expert_feedback(self, num_queries=1, eps=1e-5, **kwargs):

        choices = entropy_mp_select_expert_feedback(
            num_queries, eps,
            self.args.num_models,self.args.num_experts,
            self.model_belief, self.expert_belief,
            self.experts_already_seen,
            self.args.remaining_budget,
            self.args.expert_cost,
            self.simulator.expert_confs[:,self.simulator.sample_id],
            self.num_expert_queries,
            self.args.iterative_expert_gt,
            self.args.class_specific_belief,
            self.args.identifiable_experts,
        )

        return choices

    def check_query_expert(self,sample, eps=1e-10):
        # Must have already queried some models

        if self.args.timestep < self.args.prior_warmup_window:
            self.num_expert_queries = self.args.num_experts
            self.sample_expert_flag = True
        else:
            model_ids = list(self.models_already_seen)
            assert self.models_already_seen, \
                "No models were queried!"

            # Predictions from models
            model_preds = np.array([
                self.simulator.model_preds[m_id,sample] for
                m_id in model_ids])

            # Make a histogram of model predictions
            pred_hist, _ = np.histogram(
                model_preds,
                bins=self.args.num_classes,
                range=(0,self.args.num_classes))

            # Normalize the histogram
            confs = pred_hist/np.sum(pred_hist)

            # Get the entropy of the predictions
            entropy = ((stats.entropy(confs, base=2) /
                        np.log2(self.args.num_classes))
                        * self.args.entropy_tuning_param)
            if self.args.num_models == 1:
                entropy = (((stats.entropy(
                    self.simulator.model_confs[model_ids[0],sample]
                )/ np.log2(self.args.num_classes)) + eps)
                    *self.args.entropy_tuning_param)
                # print("Single model")
                # print(entropy)
                # sys.exit(1)

            # Check if the normalized entropy is greater than 1
            if entropy > 1: entropy=1
            elif entropy < 0: entropy=0
            assert not np.isnan(entropy),\
                "Entropy value is NaN in entropy"

            # Randomly decide whether to query z_i or not (this is oracle, could just select randomly from a human)
            # Goes back to question on if humans should be considered exchangeable
            self.num_expert_queries = np.random.binomial(
                n=self.args.num_experts, p=entropy)
            self.sample_expert_flag = self.num_expert_queries > 0
        return self.sample_expert_flag

    def select_feedback(self,sample, start=False, num_experts=None, *args, **kwargs):
        if not num_experts and self.args.single_expert: num_experts = 1
        elif not num_experts: num_experts = self.args.num_experts

        feedback_type=None; choices=[]

        # Select models
        # When finished, there will be no choices returned in list
        if self.num_model_queries > 0:
            choices = self._select_model_feedback(self.num_model_queries)
            return ( len(choices) ==0, "model", choices)

        # Select experts
        # When finished, there will be no choices returned in list
        elif self.sample_expert_flag or self.check_query_expert(sample):
            choices = self._select_expert_feedback(num_experts)
        return (len(choices) == 0, "expert", choices)


