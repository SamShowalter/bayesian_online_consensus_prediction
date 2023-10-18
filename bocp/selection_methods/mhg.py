#################################################################################
#
#             Project Title:  Multivariate Hypergeometric Selector
#             Date:           2023.07.26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import sys
import copy
import numpy as np
import torch
import scipy.stats as stats
import torch.distributions as td

from . import BaseBOCPSelector

from .utils import BELIEF_UPDATE_ROSTER, entropy_select_model_feedback, entropy_mp_update

try:
    from ..utils.misc import randargmax, torch_rand_argmax
    from ..utils.eval import has_consensus
    from ..utils.data import write_pkl
except:
    from utils.misc import randargmax, torch_rand_argmax
    from utils.eval import has_consensus
    from utils.data import write_pkl

from .prior_finset import FinSetPrior
from .prior_infset import InfSetPrior
from .prior_fixed import FixedPrior

PRIOR_ROSTER = {
    "infset": InfSetPrior,
    "finset": FinSetPrior,
    "fixed": FixedPrior,
}

def _gamma_trans_op(v,a,a0, *args):
    return a*v + a0

def softplus(v):
    return np.log(1 + np.exp(v))

def _normal_trans_op(v,a,a0,*args):
    return softplus(a*v + a0)

def _mixed_trans_op(v, a, a0, tau, eps = 1e-8):
    exps = np.exp(tau*np.log(v+eps))
    exps /= exps.sum()

    return a * exps + a0


ALPHA_OP_ROSTER = {
    "gamma": _gamma_trans_op,
    "normal": _normal_trans_op,
    "mixed": _mixed_trans_op,
}

#######################################################################
# Query By Committee selector
#######################################################################

class MultiHyperGeoSelector(BaseBOCPSelector):

    def __init__(self,args,simulator, init_weight=0):
        BaseBOCPSelector.__init__(self,args,simulator)
        self.init_weight=init_weight
        self.name = "multi_hyper_geo"
        self.labels = self.simulator.labels
        self.init_weight=init_weight
        self.dir = stats.dirichlet
        self.mult = stats.multinomial
        self.num_mc_samples = args.mhg_num_mc_samples

    def reset(self):
        if self.args.prior_method == "hybrid":
            self.prior_heur_method = self.args.mhg_hybrid_prior_heur_method
            if self.args.mhg_hybrid_prior_method == "infset":
                prior_method = "infset"
            elif self.args.mhg_hybrid_prior_method == "finset":
                prior_method = "finset"
        else:
            prior_method = self.args.prior_method
            self.prior_heur_method = self.args.prior_method
        self.prior_model = PRIOR_ROSTER[prior_method](
            self.args, self.simulator).to(self.args.device)
        self.reset_sampling()

    def reset_sampling(self):
        self.prior_model.reset_params()
        self.models_already_seen = set()
        self.experts_already_seen = set()
        self.expert_belief = np.zeros(self.args.num_experts)
        self.model_belief = np.zeros(self.args.num_models)
        self.num_model_queries = (
            np.random.rand(self.args.num_models) <=
            self.args.model_query_perc).astype(int).sum()

    # @torch.no_grad()
    def _compute_weighted_model_prior(self, sample_id, eps=1e-6):
        """
        Compute model prior such that the model belief
        and the model's predictions are all incorporated
        """

        # Get corrected prior
        model_feedback_ids = self.simulator.feedback['model']
        model_confs = self.simulator.model_confs[model_feedback_ids, sample_id]
        assert len(model_feedback_ids) > 0,\
        f"No models present in feedback for sample {sample_id} on timestep {self.args.timestep}"

        adjusted_confs = model_confs.flatten()

        self.simulator.model_data.data[self.args.timestep] = torch.from_numpy(
            adjusted_confs).to(self.args.device)

        # Compute global prior
        prior_data = self.prior_model.compute_prior()
        self.a =prior_data["a"].detach().cpu().numpy()
        self.a0 =prior_data["a0"].detach().cpu().numpy()
        self.tau = prior_data["tau"]

        if self.tau is not None:
            self.tau = self.tau.detach().cpu().numpy()

        map_func = ALPHA_OP_ROSTER[self.args.prior_dist]

        return map_func(adjusted_confs , self.a , self.a0, self.tau)

    def get_prior(self, sample):
        self.prior_params = self._compute_weighted_model_prior(sample)

    def _compute_infset_error_rate_forecast_reduction(self, num_steps=None):

        if (self.args.total_cost < 10):
            num_mc_samples = self.args.err_red_num_mc_samples*10
        else: num_mc_samples = self.args.err_red_num_mc_samples
        if not num_steps: num_steps = self.args.acc_inc_forecast_window

        accs = []
        self.acc = None
        prior_params = torch.from_numpy(self.prior_params)
        cur_step = self.expert_statistic.sum().long().item()
        for j in range(cur_step, cur_step + num_steps):
            if j == cur_step:
                H_j_C = torch.zeros((1, self.args.num_classes))
                if self.args.total_cost < 10:
                    inner_num_samples = self.args.num_mc_samples*100
                else:
                    inner_num_samples = self.args.num_mc_samples
            else:
                H_j_C = td.Multinomial(
                    total_count=j-cur_step,
                    probs=td.Dirichlet(
                        concentration=self.posterior_params
                    ).sample((num_mc_samples,)),
                ).sample()
                inner_num_samples = num_mc_samples

            H_j = H_j_C + self.expert_statistic.unsqueeze(0)

            theta = td.Dirichlet(prior_params.unsqueeze(0) + H_j).sample((inner_num_samples,))
            p_y = torch.eye(self.args.num_classes)[theta.argmax(dim=-1)].mean(dim=0)
            if self.acc is None:
                assert (len(p_y.shape) == 2 and
                        p_y.shape[0] == 1 and
                        p_y.shape[1] == self.args.num_classes),\
                    f"p_y shape is {p_y.shape}"
                self.acc = p_y.flatten()
            acc = p_y.max(dim=-1).values.mean()
            accs.append(acc.item())

        return accs

    def _compute_finset_error_rate_forecast_reduction(self, num_steps=None):
        num_mc_samples = self.args.err_red_num_mc_samples

        if ( self.args.total_cost < 10):
            num_mc_samples = self.args.err_red_num_mc_samples*10
        else: num_mc_samples = self.args.err_red_num_mc_samples
        if not num_steps: num_steps = self.args.acc_increase_forecast_window

        accs = []
        self.acc = None
        prior_params = torch.from_numpy(self.prior_params)
        cur_step = self.expert_statistic.sum().long().item()
        for j in range(cur_step, cur_step + num_steps):
            if j == cur_step:
                H_j_C = torch.zeros((1, self.args.num_classes))
                if self.args.total_cost < 10:
                    inner_num_samples = self.args.num_mc_samples*10
                else: inner_num_samples = self.args.num_mc_samples
            else:
                H_j_C = td.Multinomial(
                    total_count=j-cur_step,
                    probs=td.Dirichlet(
                        concentration=self.posterior_params
                    ).sample((num_mc_samples,)),
                ).sample()
                inner_num_samples = num_mc_samples

            H_j = H_j_C + self.expert_statistic.unsqueeze(0)

            if j == self.args.num_experts:
                H_C = torch.zeros((num_mc_samples, *H_j.shape))
            else:
                H_C = td.Multinomial(
                    total_count=self.args.num_experts - j,
                    probs=td.Dirichlet(
                        prior_params.unsqueeze(0) +
                        H_j).sample((inner_num_samples,)),
            ).sample()

            H = H_j_C + H_C

            rand_maxs = torch_rand_argmax(H, axis=-1)
            p_y = torch.eye(self.args.num_classes)[rand_maxs].mean(dim=0)
            if self.acc is None:
                assert (len(p_y.shape) == 2 and
                        p_y.shape[0] == 1 and
                        p_y.shape[1] == self.args.num_classes),\
                    f"p_y shape is {p_y.shape}"
                self.acc = p_y.flatten()
            acc = p_y.max(dim=-1).values.mean()
            accs.append(acc.item())

        return accs

    @torch.no_grad()
    def _compute_error_forecast(self):

        self.posterior_params = (torch.from_numpy(self.prior_params)
                                 + self.expert_statistic)

        num_experts_seen = self.expert_statistic.sum().long().item()
        STEP_ROSTER = {
            "err": 1,
            "err_red": min(self.args.acc_inc_forecast_window+1,
                    self.args.num_experts - num_experts_seen + 1)
        }

        steps = STEP_ROSTER[self.args.prior_heur]

        if self.args.inference_method =="finset":
            accs = self._compute_finset_error_rate_forecast_reduction(steps)
        elif self.args.inference_method == "infset":
            accs = self._compute_infset_error_rate_forecast_reduction(steps)

        if steps == 1: return 1 - accs[0]
        else:
            max_acc_inc = self._get_max_acc_increase(accs)
            return max_acc_inc

    def _get_max_acc_increase(self,accs):
        curr_acc = accs[0]

        acc_incs = []; max_inc = -float("inf")
        for i in range(1, len(accs)):
            inc = (accs[i] - curr_acc)/ i
            max_inc = max(inc, max_inc)

        return max_inc

    def _check_query_expert(self, sample):

        self.expert_statistic = self.simulator.expert_data[
            self.args.timestep].detach().cpu()
        expert_feedback_ids = self.simulator.feedback['expert']
        if has_consensus(
            self.simulator.expert_confs[:,sample,:],
            expert_feedback_ids,
            self.args.num_experts):
            self.acc = self.expert_statistic
            self.heuristic = 0
            return False        # For one step, you need to do steps = 2

        if self.args.timestep < self.args.prior_warmup_window:
            self.acc = self.expert_statistic
            self.heuristic = -1
            return True

        # Check for consensus
        self.get_prior(sample)
        self.heuristic = self._compute_heuristic()

        heur_reference = (self.args.posterior_error_rate if
            self.args.prior_heur == "err" else self.args.posterior_acc_inc)

        return (self.heuristic > heur_reference)

    def _compute_heuristic(self):
        self.total_training_iters = self.prior_model.total_training_iters
        # print(self.total_training_iters, self.prior_model.tol)
        self.tol = self.prior_model.tol
        return self._compute_error_forecast()

    def update(self, sample_id,
        feedback, pred, label, weights, gt_queried,
        *args, **kwargs):

        # Placeholder. This impacts our prior, but probably good
        # to keep things consistent
        # Basic method for updating feedback, so that
        # Baseline comparisons are done equally
        entropy_mp_update(sample_id,
            feedback, pred, label, weights, gt_queried,
            self.simulator.expert_confs, self.simulator.model_confs,
            self.expert_belief, self.model_belief,
            self.args.num_experts, self.args.num_models,
            class_specific_belief=self.args.class_specific_belief,
        )

    def _select_model_feedback(self, num_queries=1, eps=1e-5, **kwargs):

        # For now the feedback is free, so this doesn't matter much
        choices, self.num_model_queries = entropy_select_model_feedback(
            num_queries, eps,
            self.expert_belief,self.model_belief,
            self.args.num_experts, self.args.num_models,
            self.models_already_seen, self.num_model_queries,
            self.args.remaining_budget, self.args.model_cost,
            class_specific_belief=self.args.class_specific_belief,
        )
        return choices

    def _select_expert_feedback(self, sample, num_queries=1, eps=1e-5, **kwargs):
        """
        In this setup we assume experts are always exchangeable
        """
        all_ids = np.arange(self.args.num_experts)
        all_ids = list(set(all_ids) - self.experts_already_seen)
        num_queries = min(num_queries, len(all_ids))
        if (len(all_ids) == 0): return []

        # Choices should be random
        choices = np.random.choice(
            all_ids, replace=False, size=num_queries)
        if isinstance(choices,np.ndarray): choices=choices.tolist()
        elif not isinstance(choices,list): choices = [choices]
        self.experts_already_seen |= set(choices)

        # # Update the expert belief
        self.prior_model.new_data = True
        self.simulator.expert_data.data[self.args.timestep] += (
        torch.from_numpy(
            self.simulator.expert_confs[choices, sample]).sum(
                axis=0).to(self.args.device))

        return choices

    def select_feedback(self, sample,num_experts=None, *args, **kwargs):
        if not num_experts and self.args.single_expert: num_experts = 1
        elif not num_experts: num_experts = self.args.num_experts
        choices=[]
        if self.num_model_queries > 0:
            choices = self._select_model_feedback()
            return ( len(choices) ==0, "model", choices)

        if self._check_query_expert(sample):
            choices = self._select_expert_feedback(sample)

        return (len(choices) == 0, "expert", choices)


