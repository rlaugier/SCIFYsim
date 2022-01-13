#A few tools to parse config files
#This is mostly just a wrapper that includes a few defaults and tools

import numpy as np
from configparser import ConfigParser

import logging

"""
This submodules adds some extra functionnality to the ConfigParser module.

**It adds:**

* **getarray** to parse arrays
* **getdate** to parse dates
* **parse_file** to parse a file with the # for comments by default
* **list_all** To produce a list of all the keys


"""

logit = logging.getLogger(__name__)


def getarray(self, section, key, dtype=np.float64):
    """
    An extra get method to parse arrays 
    
    **Parameters:**
    
    * section   : (str) The section to get the data from
    * key       : (str) The key of the data
    * dtype     : A data type for the array conversion
    """
    logit.info("Pulling an array from config file")
    thestring = self[section][key]
    thelist = thestring.split(sep=",")
    thearray = np.array( thelist, dtype=dtype)
    return thearray
ConfigParser.getarray = getarray

def getdate(self, section, key, mode=None):
    """
    An extra get method to parse dates in the GENIE .prm format
    
    **Parameters:**
    
    * section   : (str) The section to get the data from
    * key       : (str) The key of the data
    * mode      : In case we need other formats
    """
    from astropy.time import Time
    if mode is not None:
        raise NotImplementedError("No modes implemented yet")
    else:
        logit.info("Pulling an array from config file")
        rawstring = self[section][key]
        listargs = rawstring.replace(" ", "").split(",")
        formated = listargs[0]+"-"+listargs[1]+"-"+listargs[2]+"T"\
                +listargs[3]+":"+listargs[4]+":"+listargs[5]
        logit.debug(rawstring)
        logit.debug(formated)
        thetime = Time(formated)
    return thetime
ConfigParser.getdate = getdate

def parse_file(file):
    """
    Just a quick macro to parse the config file
    
    **Parameters:**
    
    * file   : The path to a file
    """
    logit.info("Parsing a config file")
    aconfig = ConfigParser(inline_comment_prefixes="#")
    aconfig.read(file)
    return aconfig

def list_all(self):
    """
    An extra get method list all the values in the file
    """
    for section in self.sections():
        print("\n")
        print(section)
        for key in self[section].keys():
            print(key+" = ", self[section][key])
ConfigParser.list_all = list_all

