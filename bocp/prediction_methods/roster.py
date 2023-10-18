#################################################################################
#
#             Project Title:  Prediction Roster
#             Date:           2023.05.23
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from .even_weight_confs import EvenWeightPredictor
from .ftl import FTLPredictor
from .mp import MPPredictor
from .mhg import MHGPredictor


#################################################################################
#   Function-Class Declaration
#################################################################################

PREDICTION_ROSTER = {
    "even_weight": EvenWeightPredictor,
    "ftl":FTLPredictor,
    "mp":MPPredictor,
    "mhg": MHGPredictor,
}

#################################################################################
#   Main Method
#################################################################################



