#################################################################################
#
#             Project Title:  Evaluation Utilities
#             Date:           2023.04.03
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import math
import sys
import torch
import numpy as np

#######################################################################
# Check for consensus from expert histogram
#######################################################################

@torch.no_grad()
def has_consensus(expert_confs, feedback_ids, num_experts):

    # Make expert histogram
    if len(feedback_ids) == 0: return False
    expert_hist = expert_confs[feedback_ids].sum(axis=0)
    total_experts_seen = expert_hist.sum()
    rem_experts = num_experts - total_experts_seen
    # print(expert_hist)
    # print(total_experts_seen)

    # Sort to find second highest value
    expert_hist.sort()
    highest_vote, sec_highest_vote = expert_hist[::-1][:2]
    # print(expert_hist)
    # sys.exit(1)

    # Determine if number of experts left + 2nd highest value greater
    # Than the actual highest value
    # print("Consensus", (sec_highest_vote + rem_experts) < highest_vote)
    return (total_experts_seen == num_experts) or (sec_highest_vote + rem_experts) < highest_vote

#################################################################################
#   Function-Class Declaration
#################################################################################

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#######################################################################
# Average meter
#######################################################################


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if math.isnan(val) or n <1 or not isinstance(val,(float,int)): return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

