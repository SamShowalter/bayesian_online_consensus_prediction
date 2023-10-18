import sys
import os
import logging

sys.path.insert(1, os.path.abspath(os.path.join(__file__,"../../../bocp/")))
from bocp import main

logger = logging.getLogger(__name__)

main(logger,
     "imagenet16h",
     "fixed", "err", "finset")

















