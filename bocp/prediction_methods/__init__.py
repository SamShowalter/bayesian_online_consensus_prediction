#################################################################################
#
#             Project Title:  Base Prediction Method for BOCP
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

class BaseBOCPPredictor(ABC):

    """Base Class for OAMS"""

    def __init__(self,args,simulator):
        """TODO: to be defined. """
        ABC.__init__(self)
        self.args=args
        self.simulator=simulator
        self.name = "base_oams"

    def reset(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError








#################################################################################
#   Main Method
#################################################################################



