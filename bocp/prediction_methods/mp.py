#################################################################################
#
#             Project Title:  Model Picker Prediction Method for OAMS
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

class MPPredictor(BaseBOCPPredictor):

    def __init__(self, args, simulator, model_only=True):
        BaseBOCPPredictor.__init__(self,args,simulator)
        self.name = "model_picker"
        self.args=args
        self.model_only=model_only
        self.labels = self.simulator.labels

    def predict(self, feedback, sample_id, eps=1e-8, *args, **kwargs):

        confs=None; pred=None; gt_queried=False
        weights_to_save = np.zeros(self.args.num_models + self.args.num_experts)
        model_feedback = self.simulator.model_confs[feedback['model'], sample_id]
        expert_feedback =self.simulator.expert_confs[feedback['expert'], sample_id]

        if len(feedback['expert']) > 0: # Received expert feedback
            gt_queried=True;
            confs = expert_feedback.mean(axis=0)
        else: # Did not receive expert feedback
            w = self.simulator.sel_method.w
            not_model_keys = [
                m for m in range(self.args.num_models)
                if m not in feedback['model']]
            w[not_model_keys] = eps; w /= w.sum();
            chosen_model = np.random.choice(feedback['model'],p=w,size=1)
            confs = model_feedback[chosen_model].flatten()
            keys = chosen_model

        # (num_preds, k) * (num_preds, 1)
        weights_to_save = expert_feedback.sum(axis=0)
        pred = randargmax(confs)
        return confs, pred, weights_to_save, gt_queried


