#################################################################################
#
#             Project Title:  Data Utilities (splitting etc.)
#             Date:           2023.04.03
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import numpy as np
import torch
import hashlib

import math
import sys
import pickle as pkl
import json
import shutil
import yaml
import tarfile
import logging

logger = logging.getLogger(__name__)

#######################################################################
# Write and read
#######################################################################


def read_yaml(filename):
    if filename is None: return None
    with open(filename,'r') as file:
        return yaml.safe_load(file)

def write_yaml(data,filename):
    with open(filename,'w') as file:
        return yaml.dump(data,file)

def write_pkl(data,name):
    with open(f'{name}','wb') as file:
        data = pkl.dump(data,file)

def read_pkl(filename):
    with open(f'{filename}','rb') as file:
        data = pkl.load(file)
    return data

def read_json(filepath):
    return json.load(open(filepath,'r'))

def write_json(data,filepath):
    with open(filepath,'w') as file:
        json.dump(data,file)

def copy_item(source_filepath,dest_filepath):
    assert os.path.exists(source_filepath),\
        "ERROR: no object exists locally at path {}"\
        .format(source_filepath)
    shutil.copy(source_filepath, dest_filepath)

def write_log(body, key,**kwargs):
    f = open(key, 'w')
    f.write(body.getvalue())

def write_tarfile(output_filename, source_dir, contents_only=False):
    with tarfile.open(output_filename, "w:gz") as tar:
        if contents_only:
            for file_or_dir in os.listdir(source_dir):
                tar.add(os.path.join(source_dir,file_or_dir), arcname=os.path.basename(file_or_dir))
        else:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

#######################################################################
# Get cache filename
#######################################################################

def convert_to_bytes(item):
    return bytes(str(item),"utf-8")

def get_cache_filepath(args):
    filename = hashlib.sha256()
    filename.update(convert_to_bytes(args.target_model_perf))
    filename.update(convert_to_bytes(args.target_expert_perf))
    filename.update(convert_to_bytes(args.seed))
    path_name = filename.hexdigest()
    save_path = os.path.join(
        args.cache_root,
        f"{path_name}.pkl"
    )
    args.save_path = save_path


def cache_data(args,data):
    os.makedirs(args.cache_root, exist_ok=True)
    if args.cache_data and not os.path.exists(args.save_path):
        logger.info(f"Caching dataset at path {args.save_path}")
        write_pkl(data, args.save_path)

#######################################################################
# Get information for data
#######################################################################

def get_model_confs(args,data):
    model_ids = None
    if args.model_data == "synthetic": return data['model_confs']
    if args.model_ids: model_ids = args.model_ids
    if data.get("chosen_models",None):
        model_ids = data.get("chosen_models",None)
    if not model_ids: return data['model_confs']
    else: return data['model_confs'][model_ids]

def zip_expert_data(data, expert_data):
    expert_keys = [
        "expert_preds",
        "expert_perf",
        "expert_perf_per_class",
        "targets", "true_targets"]
    if data is None: return expert_data
    for key in expert_keys:
        if key in expert_data:
            data[key] = expert_data[key]
    return data

def zip_model_data(data, model_data):
    model_keys = ["model_confs",
            "model_preds","model_perf",
            "model_perf_per_class",
    ]

    if data is None: return model_data
    for key in model_keys:
        if key in model_data:
            data[key] = model_data[key]
    return data

def select_models(args,data):
    model_keys = ["model_confs",
            "model_preds","model_perf",
            "model_perf_per_class",
    ]
    model_sel_roster = {
        "perf": _select_models_by_perf,
        "random": _select_models_randomly,
        "id":_select_models_by_id,
    }

    num_available_models = data['model_preds'].shape[0]
    assert args.model_id_sel_method in model_sel_roster.keys(),\
        f"Model selection method {args.model_id_sel_method} not in roster"

    # Get model ids
    model_ids = model_sel_roster[
        args.model_id_sel_method](args,data,num_available_models)
    # print(model_ids)
    args.chosen_models = model_ids
    # print(model_ids)
    # sys.exit(1)

    # Replace data
    for key in model_keys:
        if key in data:
            data[key] = data[key][model_ids]

def zip_dict(base, added):
    for k,v in added.items():
        base[k] = v
    return base

def zip_args(base, added):
    for k,v in added.items():
        base.__dict__[k] = v
    return base

def select_model_by_perf(data, des_perf, model_ids, num_available_models):
    model_perf = data["model_perf"]
    perf_diff = np.abs(des_perf - model_perf)
    perf_sorted_model_ids = np.argsort(perf_diff)
    model_ids = set(model_ids)
    for m in perf_sorted_model_ids:
        if m not in model_ids: return m

def _select_models_by_perf(args,data, num_available_models):
    assert args.des_model_perfs is not None,\
        "Model perf selection chosen but no perfs provided"
    assert len(args.des_model_perfs) == args.num_models,\
        f"Num model perfs is {len(args.des_model_perfs)} but num models is {args.num_models}"
    model_ids = []
    for perf in args.des_model_perfs:
        model_ids.append(
            select_model_by_perf(
                data,perf, model_ids, num_available_models)
        )
    return model_ids


def _select_models_by_id(args,data, num_available_models):
    model_ids = data.get("chosen_models",None)
    if model_ids is None:
        model_ids = [args.chosen_model_id]
    assert model_ids is not None,\
        "Model ID selection method chosen but no IDs provided"
    return model_ids

def _select_models_randomly(args,data, num_available_models):
    return np.random.choice(range(args.num_models), replace = False,
                            size = args.num_models)


