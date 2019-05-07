import os
import sys
import time
import logging
from contextlib import contextmanager


"""
returns an empty list with shape (d0, d1, d2)
"""
def empty_list(d0, d1=None, d2=None):
    if d1 is None:
        return [[] for _ in range(d0)]
    elif d2 is None:
        return [[[] for _ in range(d1)] for __ in range(d0)]
    else:
        return [[[[] for _ in range(d2)] for __ in range(d1)] for ___ in range(d0)]


"""
returns a logger with std output and file output
"""
def getLogger(folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)

    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    if os.path.exists(os.path.join('logs', '{}.log'.format(name))):
        os.remove(os.path.join('logs', '{}.log'.format(name)))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\tmodel:{}\t%(message)s'.format(name))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(os.path.join(folder, '{}.log'.format(name)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(msg, logger):
    tstart = time.time()
    yield
    logger.info('%s done in %.3f seconds' % (msg, time.time() - tstart))