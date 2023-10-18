#################################################################################
#
#             Project Title:  Prior Creation for Optimization: Fixed
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

#######################################################################
# Inf Set Prior
#######################################################################


@torch.no_grad()
class FixedPrior(nn.Module):

    def __init__(self, args, simulator):
        super().__init__()
        self.args = args
        self.simulator = simulator
        self.a = torch.tensor([self.args.fixed_a_prior]*self.args.num_classes)
        self.a0 = torch.ones(self.args.num_classes)
        self.tau = torch.ones(self.args.num_classes)
        self.total_training_iters = 0
        self.tol = -1

    def reset_params(self):
        pass

    def offline_compute_prior(self):
        return self.compute_prior()

    def compute_prior(self):
        prior_params = {
            "a": self.a,
            "a0": self.a0,
            "tau":self.tau,
        }
        return prior_params
