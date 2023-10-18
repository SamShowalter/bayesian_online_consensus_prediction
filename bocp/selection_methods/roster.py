from .rand import RandomSelector
from .entropy import EntropySelector
from .mp import MPSelector
from .mhg import MultiHyperGeoSelector

#######################################################################
# Selection Roster
#######################################################################

SELECTION_ROSTER = {
    "random": RandomSelector,
    "entropy": EntropySelector,
    "mp": MPSelector,
    "mhg": MultiHyperGeoSelector,
}
