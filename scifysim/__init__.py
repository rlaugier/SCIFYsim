
import numpy as np

#from . import mol_dens
#from .mol_dens import mol_dens
from . import utilities # Utilities for computation
from . import combiners # A library of combiners
from . import injection # A classes to simulate fiber injection
from . import combiner  # A class to simulate the bihaviour of the combiner
from . import spectrograph # Simulate the behaviour of the spectrograph and camera
from . import director  # A class to rule them all
from . import sources   # A library of sources
from . import correctors # A classes to simulate OPD and chromatic compensators

from . import n_air # Manages the refractive index of air
from .n_air import *

from . import parsefile # Used to parse .ini files 
#from .parsefile import *

#from . import confconverter
from . import control_loop # Legacy from GENIEsim : builds closed loop TFs

from . import observatory as obs # Manages the geometry of the array
from . import plot_tools as pt  # Tools to facilitate the plotting

from . import map_manager
from . import analysis

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

version = "0.2.2"
