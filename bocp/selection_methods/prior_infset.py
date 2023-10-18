#################################################################################
#
#             Project Title:  Prior Creation for Optimization: InfSet
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .utils import inv_softplus

TRANS_ROSTER = {
    "softplus": F.softplus,
    "exp": torch.exp,
}

INV_TRANS_ROSTER = {
    "exp":torch.log,
    "softplus": inv_softplus,
}

def torch_mixed_trans_op(f, a, a0, tau, eps=1e-8):
    return (a.unsqueeze(0) * torch.softmax(tau *
            (f+1e-8).log(), dim=-1) + a0.unsqueeze(0))


#######################################################################
# Inf Set Prior
#######################################################################


class InfSetPrior(nn.Module):

    def __init__(self, args, simulator):
        super().__init__()

        self.args = args
        A_PRIOR_DIST_ROSTER = {

            "gamma": D.Gamma(args.prior_gamma_conc,
                    args.a0_prior_gamma_rate),

            "normal": D.Normal( args.prior_normal_mu,
                args.prior_normal_sigma),

            "mixed": D.Gamma(args.prior_gamma_conc,
                    args.prior_gamma_rate),
        }

        A0_PRIOR_DIST_ROSTER = {

            "gamma": D.Gamma(args.a0_prior_gamma_conc,
                    args.a0_prior_gamma_rate),

            "mixed": D.Gamma(args.a0_prior_gamma_conc,
                    args.a0_prior_gamma_rate),
            "normal": D.Normal( args.prior_normal_mu,
                args.prior_normal_sigma),
        }

        self.tau_prior = D.Gamma(args.prior_gamma_conc,args.prior_gamma_rate)
        self.simulator = simulator
        self.a_prior = A_PRIOR_DIST_ROSTER[args.prior_dist]
        self.a0_prior = A0_PRIOR_DIST_ROSTER[args.prior_dist]
        self.prior_lr = args.prior_lr
        self.num_mc_samples = args.num_mc_samples
        self.max_tol = args.prior_convergence_tol
        self.tol = 1; self.timestep = 0
        self.max_train_iters = args.prior_max_training_iters
        self.device = args.device
        self.convergence_check_iters = self.args.prior_convergence_check_iters
        self.recompute_iters = self.args.prior_recompute_iters
        self.window_size = self.args.prior_data_window_size

        self.map_func = TRANS_ROSTER[self.args.prior_param_mapping_func]
        self.inv_map_func = INV_TRANS_ROSTER[self.args.prior_param_mapping_func]

        self.register_buffer("num_experts_lgamma",
            torch.tensor(
                [self.args.num_experts+1]).lgamma()
        )

        self.register_buffer("model_data",
            self.simulator.model_data
        )

        self.register_buffer("expert_data",
            self.simulator.expert_data
        )

        self.reset()


    def reset(self):
        # Set initial prior values to mode of distribution
        self.new_data=False
        if self.args.prior_dist == "gamma":
            self.a = nn.Parameter(
                self.inv_map_func(torch.tensor([(self.args.prior_gamma_conc - 1)/
                    self.args.prior_gamma_rate]*self.args.num_classes,
                    device=self.args.device)))

            self.a0 = nn.Parameter(
                self.inv_map_func(torch.tensor([(self.args.a0_prior_gamma_conc - 1)/
                    self.args.a0_prior_gamma_rate]*self.args.num_classes,
                    device=self.args.device)))

        elif self.args.prior_dist =="mixed":
            self.a = nn.Parameter(
                self.inv_map_func(torch.tensor([(self.args.prior_gamma_conc - 1)/
                    self.args.prior_gamma_rate],
                    device=self.args.device)))

            if self.args.mhg_learn_bias:
                self.a0 = nn.Parameter(
                    self.inv_map_func(torch.tensor([(self.args.a0_prior_gamma_conc - 1)/
                        self.args.a0_prior_gamma_rate],
                        device=self.args.device)))
            else:
                self.a0 = self.inv_map_func(torch.tensor([(self.args.a0_prior_gamma_conc - 1)/
                        self.args.a0_prior_gamma_rate],
                        device=self.args.device))

            self.tau = nn.Parameter(
                self.inv_map_func(torch.tensor([(self.args.prior_gamma_conc - 1)/
                    self.args.prior_gamma_rate]*self.args.num_classes,
                    device=self.args.device)))

        elif self.args.prior_dist == "normal":
            self.a = nn.Parameter(
                (torch.tensor([self.args.prior_normal_mu]*self.args.num_classes,
                    device=self.args.device)))

            self.a0 = nn.Parameter(
                torch.tensor([self.args.prior_normal_mu]*self.args.num_classes,
                    device=self.args.device))

        self.opt = torch.optim.Adam(
            params=self.parameters(), lr=self.prior_lr)
        self.last_a = self.map_func(self.a.data.clone())
        self.last_a0 = self.map_func(self.a0.data.clone())
        if self.args.prior_dist == "mixed":
            self.last_tau = self.map_func(self.tau.data.clone())

    def reset_params(self):

        # Gather total votes for each sample up to current timestep
        # Votes remaining to explore for each sample
        # (timesteps, num_classes)
        self.total_training_iters = -1
        self.timestep = self.args.timestep
        self.prior_computed=False
        self.filter = (self.expert_data.sum(dim=-1) > 0)
        self.non_filtered_timesteps = self.filter.sum()
        self.skip_opt = (self.filter.sum() == 0)
        if self.skip_opt: return

        self.data_start = max(0, self.timestep - self.window_size)
        self.H_t =self.expert_data[self.filter][-self.window_size:]
        self.filtered_model_data =self.model_data[
            self.filter][-self.window_size:]
        if self.timestep == 0: return

        # Gather total votes for each sample up to current timestep
        # Votes remaining to explore for each sample
        self.data_start = max(0, self.timestep - self.window_size)
        if self.timestep == 0: return

        # Can be +1 if we want to compute the present prior
        self.f =self.model_data[self.filter][-self.window_size:]
        self.H_t =self.expert_data[self.filter][-self.window_size:]
        self.N_t = self.H_t.sum(dim=-1)
        self.remaining_N_votes = self.args.num_experts - self.N_t

    @torch.no_grad()
    def converged(self,a, a0, iteration, tau=None, at=0):
        if (iteration % self.convergence_check_iters) == 0:
            if tau is not None:
                self.tol = torch.maximum(
                        torch.maximum(
                            torch.abs(a - self.last_a).max(),
                            torch.abs(a0 - self.last_a0).max(),
                        ), torch.abs(tau - self.last_tau).max()
                )
                self.last_tau = tau.data.clone()
            else:
                self.tol = torch.maximum(
                            torch.abs(a - self.last_a).max(),
                            torch.abs(a0 - self.last_a0).max(),
                )

            self.last_a = a.data.clone()
            self.last_a0 = a0.data.clone()
            converged = (self.tol <= self.max_tol)
            return converged
        return False

    def _dirmult_loglik(self, alpha_t):
        total_alpha_t = alpha_t.sum(dim=-1)
        log_likelihood = total_alpha_t.lgamma() + (self.N_t+1.0).lgamma() - (total_alpha_t + self.N_t).lgamma()
        log_likelihood = log_likelihood + ((self.H_t + alpha_t).lgamma() - alpha_t.lgamma() - (self.H_t + 1.0).lgamma()).sum(dim=-1)
        return log_likelihood

    def compute_log_likelihood(self):

        tau = None
        # Multiply by the model predictions
        if self.args.prior_dist == "gamma":
            a = self.map_func(self.a)
            a0 = self.map_func(self.a0)
            alpha_t = (((a.unsqueeze(0)) *
                self.filtered_model_data)
                + a0.unsqueeze(0))

        elif self.args.prior_dist == "normal":
            a = self.a; a0 = self.a0
            alpha_t = F.softplus((a.unsqueeze(0)) *
                self.filtered_model_data
                + a0.unsqueeze(0))

        elif self.args.prior_dist == "mixed":
            a = self.map_func(self.a)
            a0 = self.map_func(self.a0)
            tau = self.map_func(self.tau)
            alpha_t = torch_mixed_trans_op(
                self.filtered_model_data, a, a0, tau)

        # Multiply by the importance sampling likelihood ratio, then take expectation
        data_likelihood = self._dirmult_loglik(alpha_t)

        # Get likelihood of the prior
        log_prior = self.a_prior.log_prob(a).sum()
        if self.args.prior_dist == "mixed":
            log_prior = log_prior + self.tau_prior.log_prob(tau).sum()

            if self.args.mhg_learn_bias:
                log_prior = log_prior + self.a0_prior.log_prob(a0).sum()

        # Posterior likelihood
        likelihood = (data_likelihood.mean() +
                      (log_prior.sum()/(self.non_filtered_timesteps)))

        return likelihood, {"a":a,"a0":a0, "tau":tau}

    def train(self):

        # Case where optimization is complete
        # Because we only consider previous samples
        if self.timestep == 0:
            if self.args.prior_dist == "gamma": return {
                "a": self.map_func(self.a),
                "a0": self.map_func(self.a0),
                "tau": torch.Tensor([0.0]),
            }

            elif self.args.prior_dist == "normal": return {
                "a": self.a,
                "a0": self.a0,
                "tau":torch.Tensor([0.0]),
            }

            elif self.args.prior_dist == "mixed": return {
                "a": self.map_func(self.a),
                "a0": self.map_func(self.a0),
                "tau":self.map_func(self.tau),
            }

        for i in range(self.max_train_iters):
            self.total_training_iters = (i+1)
            sample_dict = self._train_step()

            # print(sample_dict['tau'])
            if self.args.prior_dist == "mixed":
                converged =  self.converged(
                    sample_dict['a'], sample_dict['a0'], i+1,
                tau = sample_dict['tau'])
            else:
                converged =  self.converged(
                    sample_dict['a'], sample_dict['a0'], i+1)
            if converged: break

        return sample_dict

    # Compute loss, then update gradient steps
    def _train_step(self):
        self.opt.zero_grad()
        likelihood, sample_dict = self.compute_log_likelihood()

        self.loss = -likelihood
        self.loss.backward()
        self.opt.step()
        return sample_dict

    def offline_compute_prior(self):
        self.reset_params()
        return self.compute_prior()

    def compute_prior(self):
        if ((not self.skip_opt) and
            (self.new_data) and
           (not self.prior_computed) and
            (#self.timestep < self.window_size or
            self.timestep % self.recompute_iters == 0)):
            prior_params = self.train()
            # print(prior_params)
            self.prior_computed=True
            self.new_data=False
        else:
            if self.args.prior_dist == "gamma": return {
                "a": self.map_func(self.a),
                "a0": self.map_func(self.a0),
                "tau": torch.Tensor([0.0]),
            }

            elif self.args.prior_dist == "normal": return {
                "a": self.a,
                "a0": self.a0,
                "tau": torch.Tensor([0.0]),
            }

            elif self.args.prior_dist == "mixed": return {
                "a": self.map_func(self.a),
                "a0": self.map_func(self.a0),
                "tau": self.map_func(self.tau),
            }
        return prior_params



