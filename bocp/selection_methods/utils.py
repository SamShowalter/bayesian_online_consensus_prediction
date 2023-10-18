#################################################################################
#
#             Project Title:  Selection Utility Functions
#             Date:           2023.08.10
#
#################################################################################

import os
import sys
import copy

import torch
import numpy as np

try:
    from utils.misc import make_list
    from utils.eval import has_consensus
except:
    from ..utils.misc import make_list
    from ..utils.eval import has_consensus

#################################################################################
#   Module Imports
#################################################################################

#######################################################################
# Transformations
#######################################################################

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


#######################################################################
# Model Selection Step
#######################################################################



def entropy_select_model_feedback(
    num_queries, eps,
    expert_belief, model_belief,
    num_experts, num_models,
    models_already_seen, num_model_queries,
    remaining_budget, model_cost,
    class_specific_belief=False,
    *args, **kwargs,
 ):
    all_ids = np.arange(num_models)
    all_ids = list(set(all_ids) - models_already_seen)
    num_queries = min(num_queries, len(all_ids))
    if ((remaining_budget < model_cost) or
        (len(all_ids) == 0)): return []

    # Minimum total loss
    pseudo_model_probs = (
        model_belief[all_ids].max() - model_belief[all_ids]) + eps
    if class_specific_belief:
        pseudo_model_probs = pseudo_model_probs.sum(axis=-1)
    pseudo_model_probs /= pseudo_model_probs.sum()
    choices = np.random.choice(all_ids, size=num_queries,
                p=pseudo_model_probs, replace=False).tolist()
    choices = make_list(choices)
    models_already_seen |= set(choices)
    num_model_queries -= num_queries
    return choices, num_model_queries



def mp_select_model_feedback(
    num_queries, w,
    num_models,num_experts,
    models_already_seen, num_model_queries,
    remaining_budget, model_cost,
    ):
    all_ids = np.arange(num_models)
    all_ids = list(set(all_ids) - models_already_seen)
    num_queries = min(num_queries, len(all_ids))
    if ((remaining_budget < model_cost) or
        (len(all_ids) == 0)): return []

    # Model belief here is model losses
    choices = np.random.choice(
        all_ids, size=num_queries,
        p=w, replace=False).tolist()
    choice = make_list(choices)
    models_already_seen |= set(choices)
    num_model_queries -= num_queries
    return choices, num_model_queries



def entropy_mp_update(sample_id,
    feedback, pred, label, weights, gt_queried,
    expert_confs, model_confs,
    expert_belief, model_belief,
    num_experts, num_models,
    class_specific_belief=False,
    *args, **kwargs):
    if gt_queried:
        expert_updates = np.ones(num_experts)
        expert_updates[feedback['expert']] = expert_confs[
            feedback['expert'],sample_id, label]

        model_updates = np.ones(num_models)
        model_updates[feedback['model']] = model_confs[
            feedback['model'],sample_id,label]

        if not class_specific_belief:
            expert_belief += (1 - expert_updates)
            model_belief += (1 - model_updates)
        else:
            expert_belief[:,label] += (1 - expert_updates)
            model_belief[label] += (1 - model_updates)

#######################################################################
# Select Expert Feedback
#######################################################################


def entropy_mp_select_expert_feedback(
    num_queries, eps,
    num_models,num_experts,
    model_belief, expert_belief,
    experts_already_seen,
    remaining_budget, expert_cost,
    sample_expert_confs,
    max_num_expert_queries,
    iterative_expert_gt=False,
    class_specific_belief=False,
    identifiable_experts=False,
    **kwargs
 ):
    choices = []
    num_experts_seen = len(experts_already_seen)
    consensus = has_consensus(
        # H x K
        sample_expert_confs, list(experts_already_seen), num_experts)
    if (((not iterative_expert_gt) or (iterative_expert_gt and not consensus)) and
        (max_num_expert_queries > num_experts_seen)):

        all_ids = np.arange(num_experts)
        all_ids = list(set(all_ids) - experts_already_seen)
        num_queries = min(num_queries, len(all_ids))
        if ((remaining_budget < expert_cost) or
            (len(all_ids) == 0)): return []

        if not identifiable_experts:
            pseudo_expert_probs = (
                expert_belief[all_ids].max() - expert_belief[all_ids]) + eps
            if class_specific_belief:
                pseudo_expert_probs = pseudo_expert_probs.sum(dim=-1)
            pseudo_expert_probs /= pseudo_expert_probs.sum()
        else: pseudo_expert_probs = (1/len(all_ids))*np.ones(len(all_ids))

        choices = np.random.choice(
            all_ids, replace=False,
            p=pseudo_expert_probs,
            size= 1 if iterative_expert_gt else num_queries)
        choices = make_list(choices)
        experts_already_seen |= set(choices)

    return choices

#######################################################################
# Belief update roster
#######################################################################

BELIEF_UPDATE_ROSTER = {
    "entropy": entropy_mp_update,
    "mp": entropy_mp_update,
}



#################################################################################
#   Function-Class Declaration
#################################################################################



#################################################################################
#   Main Method
#################################################################################



