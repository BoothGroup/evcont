#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:13:37 2025

Logging utilities

@author: Kemal Atalar
"""

import logging
import time
from contextlib import contextmanager

# ---- Logger Setup ----
logger = logging.getLogger("timing_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():  # prevent duplicate handlers if imported multiple times
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---- Timing Context Manager ----
@contextmanager
def log_time(section_name):
    start = time.time()
    logger.info(f"Start: {section_name}")
    yield
    end = time.time()
    logger.info(f"End: {section_name} | Elapsed: {end - start:.3f} s")

# ---- Function Timing Decorator ----
def timeit(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Start: {func.__name__}")
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.info(f"End: {func.__name__} | Elapsed: {time.time() - t0:.3f} s")
        return result
    return wrapper

