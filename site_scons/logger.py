import os

try:
    import json
except ImportError as e:
    import simplejson as json

import logging.handlers

from logging import INFO, DEBUG, WARNING, ERROR, CRITICAL

structured  = logging.getLogger("structured")
tools       = logging.getLogger("Tools")

def default_strfun(tool = ""):
    def wrapper(target, source, executor):
        if tools.isEnabledFor(logging.INFO):
            tools.info("%s - %s" % (str.center(tool, 20), str(target[0])))
        if tools.isEnabledFor(logging.DEBUG):
            tools.debug("%s - %s, %s" % (tool, str([str(t) for t in target]), str([str(s) for s in source])))
    return wrapper

# LOGGER CONFIGURATION

structured.propagate = False

structured_ch = logging.FileHandler(os.environ.get('ESTUAIRE_LOGFILE', './estuaire.log'))
#structured_ch = logging.StreamHandler()
#structured_ch = logging.StreamHandler()
#structured_formatter = JSONFormatter()
structured_formatter = logging.Formatter("%(message)s")
structured_ch.setFormatter(structured_formatter)

structured.addHandler(structured_ch)


