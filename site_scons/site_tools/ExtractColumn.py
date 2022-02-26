#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = """
"""


from SCons.Script import Builder, Action
import numpy as np
import pickle

import logger

import logging

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

log = logging.getLogger('tools.ExtractColumn')

def ExtractColumn(source, target, env):
    """
    Source  :       0 - Table File
                    1 - Colunm Name (Value)

    Target  :       0 - Output
    """

    tfile = str(source[0])
    colname = source[1].value

    outfile = str(target[0])

    column = np.load(tfile).data[colname]
    if log.isEnabledFor(logging.INFO):
        log.info("Extracted %s Info <range=[%f, %f], average=%f, stdev=%f" % \
                 (colname, column.max(), column.min(), np.average(column), np.std(column)))
    pickle.dump(column, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def ExtractData(source, target, env):
    """
    :source 0:  Grid Descrition

    :target 1:  Output
    """
    gfile = str(source[0])
    grid = np.load(gfile)

    outfile = str(target[0])
    pickle.dump(grid.data, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)




def generate(env):
    env['BUILDERS']['ExtractColumn'] = Builder(action = Action(ExtractColumn,
                                                               strfunction = logger.default_strfun("Extracting Column")))

    env['BUILDERS']['ExtractData'] = Builder(action = Action(ExtractData,
                                                             strfunction = logger.default_strfun("Extracting Data")))


def exists(env):
    return 1
