#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
#

from SCons.Script import Builder, Action
import escripts

import subprocess

import numpy as np

def CheckerboardAction(target, source, env):
    """
    Scons Action
    Target :        0 - Output Grid

    Source :        0 - Shape
                    1 - Spacing
                    2 - Sigma
                    3 - BG Velocity
                    4 - Variation in Percentage of BG Velocity (optional)
    """
    shape, spacing, sigma, bgvel = [s.value for s in source[:4]]
    output = str(target[0])

    shape = ",".join(map(str, shape))

    cmd = [escripts.CHECKERBOARD, "--shape", shape, "--spacing", str(spacing),
           "--bgvel", str(bgvel), "--output", output, "--sigma", str(sigma)]

    if len(source) > 4:
        cmd += ['--percvel', str(source[4].value)]

    return subprocess.Popen(cmd).wait()

def CheckerboardLikeAction(target, source, env):
    """
    Scons Action
    Target :        0 - Output Grid

    Source :        0 - Grid
                    1 - Sigma
                    2 - BG Velocity
                    3 - Variation in Percentage of BG Velocity (optional)
    """

    grid = np.load(str(source[0]))
    sigma, bgvel = [s.value for s in source[1:3]]
    output = str(target[0])

    shape = ",".join(map(str, grid.shape))
    origin = ",".join(map(str, grid.origin))

    cmd = [escripts.CHECKERBOARD, "--shape", shape, "--spacing", str(grid.spacing),
           "--bgvel", str(bgvel), "--output", output, "--sigma", str(sigma), "--origin",
           '"%s"' % ",".join(map(str, grid.origin))]

    print cmd

    if len(source) > 4:
        cmd += ['--percvel', str(source[4].value)]

    return subprocess.Popen(cmd).wait()




def generate(env):
    env['BUILDERS']['Checkerboard'] = Builder(action = CheckerboardAction)
    env['BUILDERS']['CheckerboardLike'] = Builder(action = CheckerboardLikeAction)

def exists(env):
    return 1



