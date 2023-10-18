#################################################################################
#
#             Project Title:  Follow the Leader Prediction (used for Entropy and MP)
#                             -> Picks current best option based on model beliefs
#                             -> Alternatively, picks experts if present
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

from . import BaseBOCPPredictor
from utils.misc import randargmax
from selection_methods.utils import entropy_mp_update

logger = logging.getLogger(__name__)

#################################################################################
#   Function-Class Declaration
#################################################################################

class FTLPredictor(BaseBOCPPredictor):
    """
    Picks model with best historical accuracy on labeled data (Follow the Leader)
    If oracles are available, we default to committee vote
    """

    def __init__(self, args, simulator, model_only=True):
        BaseBOCPPredictor.__init__(self,args,simulator)
        self.name = "follow_the_leader"
        self.labels = self.simulator.labels
        self.reset()

    def update(self, sample_id,
        feedback, pred, label, weights, gt_queried, *args, **kwargs):
        # Nothing to update in this selection method
        pass

    def predict(self, feedback, sample_id, eps=1e-8, *args, **kwargs):

        confs = None; keys=None; is_gt=False
        weights_to_save = np.zeros(self.args.num_models + self.args.num_experts)

        model_keys = feedback['model']
        expert_keys = [self.args.num_models + key for key in feedback['expert']]
        model_feedback = self.simulator.model_confs[model_keys, sample_id]
        expert_feedback = self.simulator.expert_confs[
            feedback['expert'], sample_id]

        if feedback['expert']:
            is_gt=True; keys = expert_keys
            confs = expert_feedback.mean(axis=0)
            confs /= confs.sum()
        else:
            # best_models = np.argsort(self.model_belief)
            model_pseudo_probs = (
                self.simulator.sel_method.model_belief.max() -
                self.simulator.sel_method.model_belief) + eps
            not_seen_models = [
                m for m in range(self.args.num_models) if m not in model_keys]
            model_pseudo_probs[not_seen_models] = eps
            model_pseudo_probs /= model_pseudo_probs.sum()
            best_model = -1; cnt = 0
            best_model = np.random.choice(range(self.args.num_models),
                                        p=model_pseudo_probs)
            assert best_model in set(model_keys),\
                f"Sampled model {best_model} not in model keys {model_keys}"
            confs= model_feedback[best_model]; keys = best_model

        assert confs is not None,\
            "Somehow confidences did not get assigned for FTL!"
        weights_to_save = expert_feedback.sum(axis=0)
        pred = randargmax(confs)
        return confs, pred, weights_to_save, is_gt


#################################################################################
#   Main Method
#################################################################################



