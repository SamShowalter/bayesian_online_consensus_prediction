#################################################################################
#
#             Project Title:  Multivariate Hypergeometric Predictor
#             Date:           2023.10.01
#
#################################################################################

import os
import sys
import copy

import numpy as np

from utils.misc import randargmax
from . import BaseBOCPPredictor

#################################################################################
#   Function-Class Declaration
#################################################################################

class MHGPredictor(BaseBOCPPredictor):

    def __init__(self, args, simulator):
        BaseBOCPPredictor.__init__(self,args,simulator)
        self.name = "multivariate_hypergeometric"

    def predict(self, feedback, sample_id, *args, **kwargs):

        # Initialize weights to save
        weights_to_save = np.zeros(
            self.args.num_models + self.args.num_experts)

        # Get confidences base on posterior
        confs = self.simulator.sel_method.acc

        # All models and experts contribute
        expert_feedback = self.simulator.expert_confs[feedback['expert'],sample_id]
        weights_to_save = expert_feedback.sum(axis=0)

        # confs = self.simulator.sel_method.posterior_params
        confs /= confs.sum()

        # If annotators queried, use them instead of models
        pred = randargmax(confs.detach().cpu().numpy())
        return confs, pred, weights_to_save, False









#################################################################################
#   Main Method
#################################################################################



