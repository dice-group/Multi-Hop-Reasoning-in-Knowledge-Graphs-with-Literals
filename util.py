import logging
import os
import numpy as np
import random
import torch
import time


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def flatten(l): return sum(map(flatten, l), []) if isinstance(l, tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def set_logger(save_path, do_train, print_on_screen, screen_only=False):
    '''
    Write logs to console and log file
    '''
    if do_train:
        log_file = os.path.join(save_path, 'train.log')
    else:
        log_file = os.path.join(save_path, 'test.log')

    if screen_only:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a+',
        )
    if print_on_screen and not screen_only:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, epoch, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, epoch, metrics[metric]))


query_name_dict = {('a',): 'attr_exists',
                   ('e', ('r',)): '1p',
                   ('e', ('ap', 'a')): '1ap',
                   ('e', ('dp',)): '1dp',
                   ('e', ('r', 'r')): '2p',
                   (('e', ('r',)), ('ap', 'a')): '2ap',
                   ('e', ('r', 'r', 'r',)): '3p',
                   (('e', ('r', 'r',)), ('ap', 'a')): '3ap',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('dp',), ('dv', '=')): 'di',
                   (('ap', 'a'), ('v', 'f')): 'ai',
                   (('ap', 'a'), ('v', '=')): 'ai-eq',
                   (('ap', 'a'), ('v', '<')): 'ai-lt',
                   (('ap', 'a'), ('v', '>')): 'ai-gt',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f'))): '2ai',
                   (('e', ('r',)), (('ap', 'a'), ('v', 'f'))): 'pai',
                   ((('ap', 'a'), ('v', 'f')), ('r')): 'aip',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f')), ('u',)): 'au',
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())
