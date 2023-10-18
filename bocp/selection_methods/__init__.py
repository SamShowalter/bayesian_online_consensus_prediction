#################################################################################
#
#             Project Title:  Base Method for BOCP
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
import torch


#################################################################################
#   Function-Class Declaration
#################################################################################

class BaseBOCPSelector(ABC):

    """Base Class for OAMS"""

    def __init__(self,args,simulator):
        """TODO: to be defined. """
        ABC.__init__(self)
        self.args=args
        self.simulator=simulator
        self.name = "base_oams"

        # To make proposed method work
        self.heuristic=0
        self.a = np.zeros(self.args.num_classes)
        self.a0 = np.zeros(self.args.num_classes)
        self.tau = np.zeros(self.args.num_classes)
        self.total_training_iters = -1

    def reset(self):
        pass

    def reset_sampling(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_feedback(self, *args, **kwargs):
        raise NotImplementedError









#################################################################################
#   Main Method
#################################################################################



