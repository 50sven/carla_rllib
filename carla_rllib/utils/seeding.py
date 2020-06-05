"""Seeding 

This script is currently not used!

This script provides an universal seeding function.
"""

import random
import os
import numpy as np


def set_seed(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)
    os.environ['PYTHONHASHSEED'] = str(seed)
