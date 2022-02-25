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

import logger

import numpy as np
import scipy as sc
import scipy.ndimage
import pickle

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

def GaussianSmoothingAction(source, target, env):
    """
    Source :    0 - Grid Description
                1 - Sigma

    Target :    0 - Output Grid

    """
    infile = str(source[0])
    sigma = source[1].value
    outfile = str(target[0])

    grid = np.load(infile)

    grid.data = scipy.ndimage.gaussian_filter(grid.data, sigma / grid.spacing, mode = 'mirror')

    extrema = (grid.data.min(), grid.data.max())
    if logger.tools.isEnabledFor(logger.INFO):
        logger.tools.info("Range of the Smoothed %s Grid [%f, %f]" % ((outfile,) + extrema))

    pickle.dump(grid, open(outfile, 'w'), protocol = pickle.HIGHEST_PROTOCOL)

    if extrema[0] < 0:
        if logger.tools.isEnabledFor(logger.CRITICAL):
            logger.tools.critical("-- %s -- Velocity Grid Negative Value Detected" % outfile)
            logger.tools.critical("-- %s -- Choose your parameters Wisely ..." % outfile)
        return 1
    return 0



def generate(env):



    env['BUILDERS']['GaussianSmoothing'] =\
            Builder(action = Action(GaussianSmoothingAction,
                                    strfunction = logger.default_strfun("Gaussian Op.")))

def exists(env):
    return 1
