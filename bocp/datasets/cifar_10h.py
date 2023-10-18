#################################################################################
#
#             Project Title:  CIFAR-10H Dataset
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
from utils.misc import randargmax


#################################################################################
#   Function-Class Declaration
#################################################################################

class Cifar10H(object):

    def __init__(self,args):
        self.args = args
        self.data = {}
        self.max_experts = 50
        self.raw_data_path = os.path.abspath(os.path.join(__file__,
                            "../../../data/cifar-10H/cifar10h-raw.csv"))
        self.expert_preds_path = os.path.abspath(os.path.join(__file__,
                            "../../../data/cifar-10H/cifar10h_annotator_preds.pkl"))
        self.expert_ids_path = os.path.abspath(os.path.join(__file__,
                            "../../../data/cifar-10H/cifar10h_annotator_ids.pkl"))
        self.true_targets_path = os.path.abspath(os.path.join(__file__,
                            "../../../data/cifar-10H/cifar10h_true_targets.pkl"))

    def get_data(self):
        assert self.args.num_experts <= self.max_experts,\
            f"CIFAR 10H has only {self.max_experts} max experts but {self.args.num_experts} were requested"
        rand_experts = np.random.choice(range(self.max_experts),
                        replace=False,size=self.args.num_experts)
        self.args.chosen_experts = rand_experts[:self.args.num_experts]
        self.data['expert_preds'] = read_pkl(
            self.expert_preds_path)[:,rand_experts][:, :self.args.num_experts].T

        self.data['true_targets'] = read_pkl(self.true_targets_path)
        self._get_targets()
        return self.data

    def _get_targets(self):

        true_targets = self.data['true_targets']
        self.data['targets'] = stats.mode(
            self.data["expert_preds"],
            axis=0).mode.flatten()
        self.data['expert_perf'] = (
            self.data['targets'] == true_targets).astype(float).mean()
        expert_targets =self.data['targets'].flatten()
        self.data['expert_perf_per_class'] = (
            np.array([
                (expert_targets[expert_targets == i] ==
                true_targets[expert_targets==i]).astype(float).mean()
                for i in range(self.args.num_classes)
            ]))





#################################################################################
#   Main Method
#################################################################################



