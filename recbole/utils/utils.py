# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import datetime
import importlib
import os
import random

import numpy as np
import torch

from recbole.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'context_aware_recommender', 'sequential_recommender', 'knowledge_aware_recommender',
        'exlib_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:

        module_path = '.'.join(['recbole.model', submodule, model_file_name])

        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('recbole.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbole.trainer'), 'KGTrainer')
        elif model_type == ModelType.TRADITIONAL:
            return getattr(importlib.import_module('recbole.trainer'), 'TraditionalTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    # result_str = ''
    # for metric, value in result_dict.items():
    #     result_str += str(metric) + ' : ' + str(value) + '    '
    # return result_str

    result_dict = {k: round(v.real, 3) for k, v in result_dict.items()}

    # hit5="{:.3f}".format(result_dict['hit@5'])
    # hit10="{:.3f}".format(result_dict['hit@10'])
    # hit20="{:.3f}".format(result_dict['hit@20'])
    # ndcg5="{:.3f}".format(result_dict['ndcg@5'])
    # ndcg10="{:.3f}".format(result_dict['ndcg@10'])
    # ndcg20="{:.3f}".format(result_dict['ndcg@20'])
    # summary_score=0.0
    # for key, value in result_dict.items():
    #     summary_score+=value
    # summary_score=str(round(summary_score/6,3))
    # str_hit=f"hit@5: {hit5}  | hit@10: {hit10}  | hit@20: {hit20}"
    # str_ndcg=f"ndcg@5: {ndcg5} | ndcg@10: {ndcg10} | ndcg@20: {ndcg20}"
    # result_str=str_hit+'\n'+str_ndcg+'\nsummary score ---------- '+summary_score
    # return result_str

    result_str= ''
    summary_score=0
    n=0
    for k, v in result_dict.items():
        if n%2==0:
            result_str=result_str+f'{k} : {"{:.3f}".format(v)}  |  '
        n+=1
    result_str=result_str+'\n'
    n=0
    for k, v in result_dict.items():
        if n%2!=0:
            result_str=result_str+f'{k}: {"{:.3f}".format(v)}  |  '
        n+=1
        summary_score+=v**2
    summary_score=np.sqrt(summary_score/10)
    result_str=result_str+'\nsummary score -------------------------------- {:.3f}'.format(summary_score)
    return result_str


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'
