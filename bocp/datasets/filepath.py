#################################################################################
#
#             Project Title:  Filepath Dataset
#             Date:           2023.08.04
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy

import numpy as np
import torch
from scipy import stats

from utils.data import read_pkl, write_pkl

#################################################################################
#   Function-Class Declaration
#################################################################################

class FilepathDataset(object):

    def __init__(self, args):
        self.args = args
        self.data = {}

    def get_data(self):
        if self.args.data_path.split(".")[-1] == "pkl":
            self.data = read_pkl(self.args.data_path)
        elif self.args.data_path.split(".")[-1] == "pt":
            self.data = torch.load(self.args.data_path)
        elif self.args.data_path.split(".")[-1] == "pth":
            self.data = torch.load(self.args.data_path)

        if 'targets' not in self.data:
            self.data['targets'] = stats.mode(
                self.data['expert_preds'], axis=0).mode.flatten()


#################################################################################
#   Main Method
#################################################################################



