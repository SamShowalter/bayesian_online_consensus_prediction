#################################################################################
#
#             Project Title:  Summary Helper Code
#             Date:           2023-05-01
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import glob
import logging
import itertools
import pickle
import typing

import torch
import numpy as np
import pandas as pd

from .data import read_pkl, write_pkl
from .eval import AverageMeter, accuracy

logger = logging.getLogger(__name__)

#################################################################################
#   Function-Class Declaration
#################################################################################

class ResultStore(object):

    def __init__(self,args):
        self.args = args
        self.save_cols = SAVE_COLS
        self.clear()

    def clear(self):
        self.store = []

    def add_record(
        self, args,
        sel_method,
        pred_method,
        trial_id,
        timestep, sample_id,
        prediction, label,
        confidences, correct,
        orig_confidences, expert_hist,
        expert_ids,
        remaining_budget,
        total_cost,
        feedback_weights,
     ):

        high, sec_high = np.sort(expert_hist)[::-1][:2]
        is_tie = (high == sec_high)
        out_packet = {
            "trial_id":trial_id,
            "timestep":timestep,
            "sample_id": sample_id,
            "prediction": prediction,
            "label":label.item(),
            # "remaining_budget":remaining_budget,
            "total_cost":total_cost,
            "confidence": confidences[prediction],
            "correct": correct,
            "heuristic": sel_method.heuristic,
            "prior_training_iters": sel_method.total_training_iters,
            "is_tie": int(is_tie),
        }

        feedback_weights = feedback_weights.tolist()

        out_packet = {**out_packet, **{
            f"mc_{i}": confidences[i]
            for i in range(self.args.num_classes)
        }}

        out_packet = {**out_packet, **{
            f"ec_{i}": expert_hist[i]
            for i in range(self.args.num_classes)
        }}

        out_packet = {**out_packet, **{
            f"omc_{i}": orig_confidences[i]
            for i in range(self.args.num_classes)
        }}

        if args.prior_dist != "mixed":
            out_packet = {**out_packet, **{
                f"a_{i}": sel_method.a[i] for
                i in range(self.args.num_classes)
            }}
        else:
            out_packet = {**out_packet, **{
                f"a": sel_method.a[0]
            }}
            out_packet = {**out_packet, **{
                f"tau_{i}": sel_method.tau[i] for
                i in range(self.args.num_classes)
            }}


        out_packet = {**out_packet, **{
            f"a0": sel_method.a0[0]
        }}

        out_packet = {**out_packet, **{
            f"ef_{i}": 1 if i in expert_ids else 0
            for i in range(self.args.num_experts)
        }}

        out_packet = {**out_packet, **{
            f"m{i}_id":self.args.chosen_models[i]
            for i in range(self.args.num_models)
        }}

        out_packet = {**out_packet, **{
            f"e{i}_id":self.args.chosen_experts[i]
            for i in range(self.args.num_experts)
        }}


        # out_packet = {**out_packet, **{
        #     f"a0_{i}": sel_method.a0[i] for
        #     i in range(self.args.num_classes)
        # }}

        out_packet = {k: to_item(v) for k,v
                       in out_packet.items()}

        # for col in HYPERPARAM_COLS:
        #     out_packet[col] = vars(args)[col]

        self.store.append(out_packet)


    def save(self):
        model_cols = [f"mc_{i}" for i in range(self.args.num_classes)]
        expert_cols = [f"ec_{i}" for i in range(self.args.num_classes)]
        orig_model_cols = [f"omc_{i}" for i in range(self.args.num_classes)]
        expert_flag_cols = [f"ef_{i}" for i in range(self.args.num_experts)]
        model_id_cols = [f"m{i}_id" for i in range(self.args.num_models)]
        expert_id_cols = [f"e{i}_id" for i in range(self.args.num_experts)]
        tau_cols = ([f"tau_{i}" for i in range(self.args.num_classes)]
                if self.args.prior_dist == "mixed" else [])

        if self.args.prior_dist != "mixed":
            a_cols = [f"a_{i}" for i in range(self.args.num_classes)]
        else: a_cols = ["a"]
        # a0_cols = [f"a0_{i}" for i in range(self.args.num_classes)]
        a0_cols = [f"a0"]
        df = pd.DataFrame(
            self.store,
            columns=SAVE_COLS + model_cols + expert_cols + orig_model_cols+  a_cols + tau_cols + a0_cols+ expert_flag_cols + model_id_cols + expert_id_cols,
            # columns=SAVE_COLS+ model_id_cols + expert_id_cols + model_cols + expert_cols +  a_cols + tau_cols + a0_cols,
        )

        save_path = os.path.join(
            self.args.out_root, self.args.experiment_name)
        os.makedirs(save_path,exist_ok=True)
        # write_pkl(df, os.path.join(save_path,"results.pkl"))
        logger.info(f"Saving results to {save_path}")
        df.to_csv(os.path.join(save_path,"results.csv"),index=None)


def to_item(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return v

#######################################################################
# Save Columns
#######################################################################

SAVE_COLS = [
    "trial_id", "timestep", "sample_id", "prediction",
    "label","confidence","correct", "total_cost",
    "heuristic","is_tie", "prior_training_iters",
]

# HYPERPARAM_COLS = [
#     "num_model_queries", "num_expert_queries", # number of selected models and experts
#     "qbc_entropy_tuning_param", # Scale the entropy up and down
#     "mp_tuning_param", # Scale the entropy up and down
# ]


