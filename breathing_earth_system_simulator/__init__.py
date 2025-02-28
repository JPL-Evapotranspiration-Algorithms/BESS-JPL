from .FLiESANN import *
from .breathing_earth_system_simulator import *

from os.path import join, dirname, abspath

with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read()

__version__ = version
__author__ = "Gregory H. Halverson"

