
import numpy as np

#from . import mol_dens
#from .mol_dens import mol_dens
from . import n_air
from .n_air import *

from . import parsefile
#from .parsefile import *

#from . import confconverter
from . import control_loop

from . import observatory as obs
from . import plot_tools as pt

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#print("Testing the file parsing")
parent = pathlib.Path(__file__).parent.absolute()
#print("The parent dir", parent)
log_config_file = parent/"config/logger.ini"
#print("The file to grab", log_config_file)
logger_config = parsefile.parse_file(log_config_file)
#print(logger_config.get("handlers", "keys"))

import logging
import logging.config

logging.config.fileConfig(fname=log_config_file, disable_existing_loggers=False)
logit = logging.getLogger(__name__)

logit.info("SCIFYsim successfully loaded")





