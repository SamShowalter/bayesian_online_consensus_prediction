#################################################################################
#
#             Project Title:  Random Method for BOCP Selection
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
import logging

import numpy as np
import torch

from . import BaseBOCPSelector
from utils.eval import has_consensus

logger = logging.getLogger(__name__)

#################################################################################
#   Function-Class Declaration
#################################################################################

class RandomSelector(BaseBOCPSelector):

    """Random Feedback Selector"""

    def __init__(self,args,simulator):
        BaseBOCPSelector.__init__(self,args,simulator)
        self.name = "random"
        self.reset()

    def reset(self):
        self.reset_sampling()

    def reset_sampling(self, eps=1e-5):

        # self.budget_left=self.args.remaining_budget/(self.args.seq_len - self.args.curr_timestep)
        self.model_query_perc = self.args.model_query_perc
        self.expert_query_perc = self.args.rand_expert_query_perc
        self.num_expert_queries = None
        self.models_remaining=True
        self.experts_remaining=True
        self.models_already_seen = set()
        self.experts_already_seen = set()

    def _select_feedback(self, *args, **kwargs):
        feedback_type = "model"

        # Query models all at once
        if self.model_query_perc > 0 and self.models_remaining:
            self.num_model_queries = (
                np.random.rand(self.args.num_models) <=
                self.model_query_perc).astype(int).sum()

            # Case where no experts are taken
            if self.num_model_queries == 0:
                return "model", None

            choices = np.random.choice(
                range(self.args.num_models),
                size=self.num_model_queries)
            self.models_already_seen = set(choices)
            self.models_remaining=False # Random selects in one shot
            return "model", choices

        # Query experts
        elif (self.experts_remaining and ((self.num_expert_queries is None) or
            (self.num_expert_queries > len(self.experts_already_seen)))):
            if self.args.timestep < self.args.prior_warmup_window:
                self.num_expert_queries = 0
                expert_choices = list(range(self.args.num_experts))
                self.experts_remaining = 0

            else:
                self.num_expert_queries = (
                    np.random.rand(self.args.num_experts) <=
                    self.expert_query_perc).astype(int).sum()

                if self.num_expert_queries == 0:
                    return "expert", None

                # Get expert IDs
                expert_ids = [ c for c  in
                    range(self.args.num_experts)
                    if c not in self.experts_already_seen
                ]
                expert_choices = np.random.choice(
                    expert_ids,
                    size= 1 if self.args.iterative_expert_gt
                    else self.num_expert_queries)
                self.experts_already_seen |= set(expert_choices)
                self.experts_remaining= (
                    not has_consensus(
                        self.simulator.expert_confs[:,self.simulator.sample_id],
                        list(self.experts_already_seen), self.args.num_experts)
                    if self.args.iterative_expert_gt
                    else False)
            return "expert", expert_choices

        # Terminate sampling
        else:
            return None, None

    def select_feedback(self,sample, start=False, *args, **kwargs):

        feedback_type, choice = self._select_feedback(
            self.args.num_models,
            self.args.num_experts,
        )
        return (
            (choice is None, feedback_type, choice))












#################################################################################
#   Main Method
#################################################################################



