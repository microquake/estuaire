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

import eikonal.data

import logger

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

def HomogenousGridAction(source, target, env):
    """
    Source :        0 - Shape (Value)
                    1 - Fill (Value)

    Target :        2 - Grid File
    """

    shape, value, origin, spacing = [s.value for s in source[:4]]
    gridfile = str(target[0])

    grid = np.empty(shape, dtype = 'float')
    grid.fill(value)
    pickle.dump( grid, open(gridfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def HomogenousGridFromGridDescription(source, target, env):
    """
    :source 0:      Grid Description (Value)

    :target 0:      Output Grid File
    """
    grid_description, fill_value = [s.value for s in source]
    outfile = str(target[0])

    grid = grid_description.gen_homogeneous_grid(fill_value)
    pickle.dump(grid, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def HomogenousGridLike(source, target, env):
    """
    """
    grid = np.load(str(source[0]))
    fill_value = source[1].value
    outfile = str(target[0])

    outgrid = eikonal.data.EKImageData(grid.shape, grid.spacing, origin = grid.origin)
    outgrid.data.fill(fill_value)

    pickle.dump(outgrid, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)




def generate(env):
    env['BUILDERS']['HomogenousGrid'] = \
            Builder(action = Action(HomogenousGridAction,
                                    strfunction = logger.default_strfun("Homogeneous Grid")))

    env['BUILDERS']['HomogenousGrid2'] = \
            Builder(action = Action(HomogenousGridFromGridDescription,
                                    strfunction = logger.default_strfun("Homogeneous Grid")))

    env['BUILDERS']['HomogenousGridLike'] = \
            Builder(action = Action(HomogenousGridLike,
                                    strfunction = logger.default_strfun("Homogenous Grid")))




def exists(env):
    return 1
