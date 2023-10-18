#################################################################################
#
#             Project Title:  Experts for Bandit experiments
#             Date:           2023-05-11
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import torch
import numpy as np
import copy


#################################################################################
# Function
#################################################################################

class Expert(object):

    """Expert Labeler (Human)"""

    def __init__(self, labels, probs, num_classes=10, verbose=True):
        self.probs= probs
        self.verbose=verbose
        self.labels=labels
        self.num_classes= num_classes
        self.make_preds()

    def make_preds(self):
        self.preds = copy.deepcopy(self.labels)

        for i in range(self.num_classes):
            rand_preds = np.random.randint(
                self.num_classes, self.labels.shape)
            rand_probs = np.random.rand_like(self.labels)
            # print(self.probs[i], (rand_probs > self.probs[i]).float().mean())
            self.preds[(rand_probs > self.probs[i])
                       & (self.labels == i)] = rand_preds[
                           (rand_probs > self.probs[i]) &
                            (self.labels == i)]

        if self.verbose:
            acc = (self.preds == self.labels).mean()*100
            print(f"Expert probs: {[round(p*100,2) for p in self.probs.tolist()]}")
            print("Expert accuracy: {:.2f}".format(acc))

