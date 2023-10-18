#################################################################################
#
#             Project Title:  Base Prediction Method for OAMS
#             Date:           2023.05.22
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from abc import ABC, abstractmethod
import os
import sys
import copy

import numpy as np

from utils.misc import randargmax
from . import BaseBOCPPredictor

#################################################################################
#   Function-Class Declaration
#################################################################################

class EvenWeightPredictor(BaseBOCPPredictor):

    def __init__(self, args, simulator):
        BaseBOCPPredictor.__init__(self,args,simulator)
        self.name = "even_weight"
        self.ew = self.args.expert_weight
        self.mw = self.args.model_weight

    def predict(self, feedback, sample_id, *args, **kwargs):

        weights_to_save = np.zeros(self.args.num_models + self.args.num_experts)
        confs = self.simulator.model_confs[feedback['model'],sample_id]
        expert_feedback = self.simulator.expert_confs[feedback['expert'],sample_id]
        if len(feedback['expert']) > 0:
            if len(expert_feedback.shape) > 1:
                confs = expert_feedback.mean(axis=0)

        else:
            if len(confs.shape) > 1:
                confs = confs.mean(axis=0)

        weights_to_save = expert_feedback.sum(axis=0)
        # (num_preds, k) * (num_preds, 1)
        confs/=confs.sum()
        pred = randargmax(confs)
        return confs, pred, weights_to_save,(len(feedback['expert'])==self.args.num_experts)









#################################################################################
#   Main Method
#################################################################################



