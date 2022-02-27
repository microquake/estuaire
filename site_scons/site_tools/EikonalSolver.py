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

import os
import subprocess
import numpy as np
import pickle

import logger
import escripts


def np_load(*args, **kwargs):
    return lambda *a, **k: np.load(*a, allow_pickle=True, **k)


def EikonalSolver2(source, target, env):
    """
    :source 0: Velocity Grid
    :source 1: TravelTime File

    :target 0: Arrival Grid

    """
    velocityfile = str(source[0])
    vgrid = pickle.load(open(velocityfile, 'rb'))

    tttable = np_load(str(source[1]))
    seed = tttable.station_row['position']

    strseed = [str(s) for sh, s in zip(vgrid.shape, seed)]

    cmd = [escripts.EIKONAL_SOLVER, "--velocity", velocityfile, "--seeds",
           '"%s"' % ",".join(strseed), "--output",
           str(target[0])]

    return subprocess.Popen(cmd).wait()


def EikonalSolver(source, target, env):
    """
    Source :        0 - Velocity
                    1 - Seed (Value)

    Target :        0 - TravelTime Grid
    """
    velocityfile = str(source[0])
    seed = source[1]

    strseed = [str(s) for s in seed]

    script = os.path.join(env['TOMOGRAPHY_SCRIPT_DIR'], scriptfile)

    cmd = "python %s --velocity=%s --seeds=%s --spacing=%d --output=%s" % \
            (script, velocityfile, ",".join(strseed), spacing, str(target[0]))
    result = os.popen(cmd)

def generate(env):
    env['BUILDERS']['EikonalSolver'] = \
            Builder(action = Action(EikonalSolver, strfunction = logger.default_strfun(tool = "Eikonal")))

    EikonalSolverAction = Action(EikonalSolver2, strfunction = logger.default_strfun(tool = "Eikonal"))
    env['BUILDERS']['EikonalSolver2'] = \
            Builder(action = EikonalSolverAction)

def exists(env):
    return 1
