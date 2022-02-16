#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = """
"""

from SCons.Script import Builder

import pickle
import os

from eikonal import linear

def GridDifference(source, target, env):
    """
    Source :        0 - Grid Shape (Value)
                    1 - 1st Derivative Weight (Value)
                    2 - 2nd Derivative Weight (Value)
                    3 - Spacing (Value)

    Target :        0 - Output file
    """
    eta = [1, 1]

    eta[:] = source[1].value

    if len(source) > 2:
        spacing = source[2].value
        eta[0] /= spacing
        eta[1] /= spacing
    shape = source[0].value
    output = str(target[0])

    d, dd = linear.linear_derivative_op(shape)

    pickle.dump(d * eta[0] + 0.5 * eta[1] * dd, open(output, 'w'),
                protocol = pickle.HIGHEST_PROTOCOL)


def generate(env):
    env['BUILDERS']['GridDifference'] = Builder(action = GridDifference)

def exists(env):
    return 1
