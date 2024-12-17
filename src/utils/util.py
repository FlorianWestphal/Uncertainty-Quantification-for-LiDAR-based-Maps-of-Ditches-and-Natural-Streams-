import logging
import os
import numpy as np


def select_from_index(data, index):
    '''Select the values from data which were indicated by the index

    Parameters
    ----------
    data : Data to select from [C, K_n, ..., K_1]
    index : Array of index values [K_n, ..., K_1] with indices ranging from 0
    to C

    Returns
    -------
    Selected values [K_n, ..., K_1]

    '''
    shape = index.shape
    idx = index.flatten().astype(np.uint8)
    entries = len(idx)
    cls_num = data.shape[0]

    selected = data.reshape((cls_num, -1))[idx, np.arange(entries)]

    return selected.reshape(shape)


def enable_logging(log_path, file_name):
    '''Setup logging

    Parameters
    ----------
    log_path : Path to store log file
    file_name : Log file name

    '''
    # setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
                filename=os.path.join(log_path, file_name),
                filemode='w', level=logging.INFO,
                datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
