import itertools
import copy

"""
this code is taken from https://github.com/cesarali/Tyche/blob/develop/src/tyche/utils/helper.py
"""
def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if '.' in prefix:
                prefix = prefix.rsplit('.', 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if '.' in prefix:
                        prefix = prefix.rsplit('.', 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params

def expand_config(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d['name']
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "_" + j[0] + "_" + str(j[1])
            d['name'] = name.replace('.args.', "_")
        d = convert_tuples_2_list(d)
        cv_params.append(d)
    if not cv_params:
        return [params] * params['num_runs']

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p['num_runs']
    return gs_params

def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if not key in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if not key in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]



def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg