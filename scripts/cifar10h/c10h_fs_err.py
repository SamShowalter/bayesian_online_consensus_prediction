import sys
import os
import logging

sys.path.insert(1, os.path.abspath(os.path.join(__file__,"../../../bocp/")))
from bocp import main

logger = logging.getLogger(__name__)

# CIFAR-10H FinSet Err
main(logger,
     os.path.join(__file__,"../../../data/cifar-10/cifar10/cifar10-fixmatch-rs1-s40.pkl"),
     "finset", "err")

