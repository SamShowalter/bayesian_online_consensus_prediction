#################################################################################
#
#             Project Title:  Imagenet 16H Dataset
#             Date:           2023.08.05
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import copy
import sys

import numpy as np
from scipy import stats
from utils.data import read_pkl


#################################################################################
#   Function-Class Declaration
#################################################################################

class Imagenet16hDistShift(object):

    def __init__(self,args):
        self.args = args
        self.data = {}
        self.max_experts = 6
        self.max_models = 5
        self.num_unique_samples = 1200
        self.imagenet_clean_experts = self.args.imagenet_clean_experts
        self.raw_data_path = os.path.abspath(os.path.join(__file__,
                            "../../../data/imagenet-16H/imagenet16h_dist_shift.pkl"))

    def get_data(self):
        assert self.args.num_experts <= self.max_experts,\
            f"Imagenet 16H has only {self.max_experts} max experts but {self.args.num_experts} were requested"
        self.data = read_pkl(self.raw_data_path)
        rand_experts = np.random.choice(range(self.max_experts),
                                        replace=False,size=self.args.num_experts)
        self.args.chosen_experts = rand_experts[:self.args.num_experts]
        expert_key = 'expert_preds_clean' if self.imagenet_clean_experts else 'expert_preds_dirty'
        self.data['model_confs'] /= self.data['model_confs'].sum(axis=-1)[...,None]
        assert ((self.data['model_confs'].sum(axis=-1) - 1.0) < 1e-8).all(),\
            "Model confidences did not sum to one"
        self.data['expert_preds'] = self.data[expert_key][
            rand_experts[:self.args.num_experts]]
        self._get_targets()
        self._get_model_perf()
        return self.data

    def _get_model_perf(self):

        true_targets = self.data['true_targets']
        model_preds = self.data['model_preds']
        self.data['model_perf'] = (
            true_targets == model_preds).astype(float).mean(axis=-1)
        self.data['pre_model_perf'] = (
            true_targets[:self.num_unique_samples] == model_preds[:,:self.num_unique_samples]).astype(float).mean(axis=-1)
        self.data['post_model_perf'] = (
            true_targets[:self.num_unique_samples] == model_preds[:,self.num_unique_samples:]).astype(float).mean(axis=-1)

        self.data['model_perf_per_class'] = (
            np.stack([np.array([
                (model_preds[j,model_preds[j] == i] ==
                 true_targets[model_preds[j] == i]).astype(float).mean()
                for i in range(self.args.num_classes)
            ]) for j in range(self.max_models)]))


    def _get_targets(self):

        true_targets = self.data['true_targets']
        self.data['targets'] = stats.mode(
            self.data["expert_preds"],
            axis=0).mode.flatten()
        self.data['expert_perf'] = (
            self.data['targets'] == true_targets).astype(float).mean()

        self.data['pre_expert_perf'] = (
            self.data['targets'] == true_targets).astype(float).mean()
        expert_targets =self.data['targets'].flatten()
        expert_data =self.data['expert_preds']

        self.data['expert_perf_per_class'] = (
            np.stack([np.array([
                (expert_data[j,expert_data[j] == i] ==
                 true_targets[expert_data[j] == i]).astype(float).mean()
                for i in range(self.args.num_classes)
            ]) for j in range(self.max_experts)]))





#################################################################################
#   Main Method
#################################################################################



