import csv, gc, gzip, os, pickle, shutil, sys, warnings, yaml, io, subprocess
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import abc, collections, hashlib, itertools, json, operator, pathlib
import mimetypes, inspect, typing, functools, importlib, weakref
import html, re, requests, tarfile, numbers, tempfile, bz2

from abc import abstractmethod
from collections import abc,  Counter, defaultdict, namedtuple, OrderedDict
from collections.abc import Iterable
import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field, InitVar
from enum import Enum, IntEnum
from functools import partial, reduce
from pdb import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from warnings import warn
from contextlib import contextmanager
from fastprogress.fastprogress import MasterBar, ProgressBar
from matplotlib.patches import Patch
from pandas import Series, DataFrame
from io import BufferedWriter, BytesIO

"""
import pkg_resources
pkg_resources.require("fastprogress>=0.1.19")
from fastprogress.fastprogress import master_bar, progress_bar

#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None

def have_min_pkg_version(package, version):
    "Check whether we have at least `version` of `package`. Returns True on success, False otherwise."
    try:
        pkg_resources.require(f"{package}>={version}")
        return True
    except:
        return False
"""
import importlib
from importlib.metadata import version, PackageNotFoundError

# Verifica versione minima di fastprogress
try:
    if tuple(map(int, version("fastprogress").split("."))) < (0, 1, 19):
        raise RuntimeError("fastprogress>=0.1.19 is required")
except PackageNotFoundError:
    raise RuntimeError("fastprogress is not installed")

from fastprogress.fastprogress import master_bar, progress_bar

# for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace


def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def have_min_pkg_version(package: str, min_version: str) -> bool:
    "Check whether we have at least `min_version` of `package`. Returns True on success, False otherwise."
    try:
        pkg_version = version(package)
    except PackageNotFoundError:
        return False
    return tuple(map(int, pkg_version.split("."))) >= tuple(map(int, min_version.split(".")))
