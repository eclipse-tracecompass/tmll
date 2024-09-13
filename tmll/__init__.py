import os
import sys

# Dynamically add the tmll/tsp submodule to the sys.path
current_dir = os.path.dirname(__file__)
tsp_dir = os.path.join(current_dir, 'tsp')
sys.path.insert(0, tsp_dir)

from .tmll_client import TMLLClient

from .common import *
from .ml import *
from .utils import *
from .services import *

# For better user experience
from .ml import modules