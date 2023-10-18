import itertools

def make_config_combs(dynamic_dict, sel_pred_methods=None,
                      model_params = None, model_param_lkp=None):
    permutation_dicts = []
    if sel_pred_methods:
        for sel_method, pred_method in sel_pred_methods:
            dynamic_dict["sel_method"] = [sel_method]
            dynamic_dict["pred_method"] = [pred_method]
            sel_method_param_key = model_param_lkp[sel_method]
            dynamic_dict[sel_method_param_key] = model_params[
                sel_method_param_key]
            permutation_dicts += _make_config_combs(dynamic_dict)
            del dynamic_dict[sel_method_param_key]

    else: permutation_dicts += _make_config_combs(dynamic_dict)
    return permutation_dicts

def _make_config_combs(dynamic_dict):
    keys, values = zip(*dynamic_dict.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutation_dicts






