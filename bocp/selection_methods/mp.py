#################################################################################
#
#             Project Title:  Model Picker Selector
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import sys
import copy
import numpy as np
import scipy.stats as stats

from . import BaseBOCPSelector
from .utils import entropy_mp_update, mp_select_model_feedback, entropy_mp_select_expert_feedback

#######################################################################
# Query By Committee selector
#######################################################################

class MPSelector(BaseBOCPSelector):

    def __init__(self,args,simulator, init_weight=0):
        BaseBOCPSelector.__init__(self,args,simulator)
        self.init_weight=init_weight
        self.name = "model_picker"
        self.w = None; self.args=args
        self._make_model_expert_belief()
        self.labels = self.simulator.labels
        self.dim = self.args.num_models
        self.eta = np.sqrt(np.log(self.dim)/2)
        if not self.args.identifiable_experts:
            self.dim += self.args.num_experts

    def _make_model_expert_belief(self):
        if not self.args.class_specific_belief:
            self.model_belief = self.init_weight*np.ones(self.args.num_models)
            self.expert_belief = self.init_weight*np.ones(self.args.num_experts)
        else:
            self.model_belief = self.init_weight*np.ones(
                (self.args.num_models, self.args.num_classes))
            self.expert_belief = self.init_weight*np.ones(
                (self.args.num_experts, self.args.num_classes))

    def _calculate_eta(self):
        return np.sqrt(np.log(self.dim)/(self.args.curr_timestep+1))

    def reset_sampling(self):
        self.models_already_seen = set()
        self.experts_already_seen = set()
        self.sample_expert_flag = False
        self.num_expert_queries = 0
        self.num_model_queries = (
            np.random.rand(self.args.num_models) <=
            self.args.model_query_perc).astype(int).sum()

    def reset(self):
        self._make_model_expert_belief()
        self.eta = np.sqrt(np.log(self.dim)/2)
        self.reset_sampling()

    def _calculate_variance(self,preds, sample_id, eps=1e-10):
        all_losses = np.stack([
            preds == i for i in range(self.args.num_classes)
        ]).astype(float)
        bernoulli = (self.w[self.model_ids].dot(all_losses.T)
                     *(1-self.w[self.model_ids].dot(all_losses.T)))
        assert ((len(bernoulli.shape) == 1) and
                (bernoulli.shape[0] == self.args.num_classes)),\
            f"Bernoulli variable shape was {bernoulli.shape}"
        bernoulli_max = bernoulli.max()
        if bernoulli_max == 0 and self.args.num_models == 1:
            bernoulli_max = (1 - self.simulator.model_confs[
                self.model_ids[0],sample_id].max())

        return self.args.mp_tuning_param*(bernoulli_max + eps)

    def _update_w(self):
        "Not sure how to update this when there are class-specific beliefs"
        self.eta = self._calculate_eta()
        if self.args.class_specific_belief:
            model_belief = self.model_belief.sum(axis=-1)
        else: model_belief = self.model_belief
        self.w = np.exp(-self.eta*model_belief)
        self.w /= self.w.sum()

    def update(self, sample_id,
        feedback, pred, label, weights, gt_queried, *args, **kwargs):

        entropy_mp_update(sample_id,
                feedback, pred, label, weights, gt_queried,
                self.simulator.expert_confs,
                self.simulator.model_confs,
                self.expert_belief, self.model_belief,
                self.args.num_experts, self.args.num_models,
                self.args.class_specific_belief,
            )

    def get_query_prob(self, sample_id):
        self._update_w()
        if not self.models_already_seen: self.q_t = 1; return 1
        self.model_ids = list(self.models_already_seen)
        confs = self.simulator.model_confs[self.model_ids,sample_id]
        self.preds = confs.argmax(axis=-1)
        assert self.preds.shape[0] == len(self.model_ids),\
            "Have different shapes for predictions and models in MP"
        var = self._calculate_variance(self.preds, sample_id)
        self.q_t = 0 if var == 0 else max(self.eta,var)
        return self.q_t

    def check_query_expert(self, sample_id):
        if self.args.timestep < self.args.prior_warmup_window:
            self.num_expert_queries = self.args.num_experts
            self.sample_expert_flag = True
        else:
            q_t = self.get_query_prob(sample_id)
            if q_t < 0: q_t=0
            if q_t > 1: q_t=1
            self.num_expert_queries = np.random.binomial(
                self.args.num_experts,p=q_t)
            self.sample_expert_flag=self.num_expert_queries > 0
        return self.sample_expert_flag

    def _select_model_feedback(self, num_queries=1, *args, **kwargs):

        self._update_w()

        choices, self.num_model_queries = mp_select_model_feedback(
            num_queries, self.w,
            self.args.num_models,self.args.num_experts,
            self.models_already_seen, self.num_model_queries,
            self.args.remaining_budget, self.args.model_cost,
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

    def select_feedback(self,sample, start=False,num_experts=None, *args, **kwargs):
        if not num_experts and self.args.single_expert: num_experts = 1
        elif not num_experts: num_experts = self.args.num_experts
        choices=[]
        if self.num_model_queries > 0:
            choices = self._select_model_feedback(self.num_model_queries)
            return ( len(choices) ==0, "model", choices)
        elif self.sample_expert_flag or self.check_query_expert(sample):
            choices = self._select_expert_feedback(num_experts)
        return (len(choices) == 0, "expert", choices)


