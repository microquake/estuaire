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
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle


def ResidualHistogram(source, target, env):
    """
    Source          0 - Residual File (A file)
                    1 - nbins (Value)
                    2 - Boundaries (Value)

    Target          0 - Graph File (Image)
    """
    plt.figure(0)
    kw = {}
    if len(source) > 1:
        kw['bins'] = source[1].value
    if len(source) > 2:
        kw['range'] = source[2].value

    residuals = pickle.load(open(str(source[0]), 'rb'))['R']

    plt.hist(residuals, **kw)
    plt.savefig(str(target[0]))

def generate(env):
    env['BUILDERS']['ResidualHistogram'] = Builder(action = ResidualHistogram)


def exists(env):
    return 1
