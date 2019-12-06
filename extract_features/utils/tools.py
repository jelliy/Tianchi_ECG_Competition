"""
tools.py
-----------
This module provides a class and methods for processing ECG data.
Implemented code assumes a single-channel lead ECG signal.
:copyright: (c) 2017 by Goodfellow Analytics
-----------
By: Sebastian D. Goodfellow, Ph.D.
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np


def shannon_entropy(array):
    return -sum(np.power(array, 2) * np.log(np.spacing(1) + np.power(array, 2)))
