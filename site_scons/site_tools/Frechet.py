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

import logger
import escripts


def Sensivity(source, target, env):
    """
    :source 0: Velocity Grid
    :source 1: Arrival Grid
    :source 2: TT FIle
    :source 3: Grid ID
    :source 4: Raytracing Step (Optional)

    :target 0: Output File
    """

    velocity, arrival, ttfile = [str(s) for s in source[:3]]
    grid_id = source[3].value

    output = str(target[0])

    cmd = [escripts.FRECHET_DERIVATIVES, "--arrival", arrival, "--velocity",
           velocity, "--grid_id", grid_id, "--output", output,  "--traveltime",
           ttfile]

    if len(source) > 4:
        cmd.extend(["--h", source[4].value])

    return subprocess.Popen(cmd).wait()


def CrossSensivity(source, target, env):
    """
    :source 0: Velocity Grid
    :source 1: Standard Velocity Grid
    :source 2: Ray Path
    :source 3: Grid ID

    :target 0: Output file
    """
    velocity, standard_velocity, raypath = [str(s) for s in source[:3]]

    output = str(target[0])
    cmd = [escripts.CROSS_SENSIVITY, "--standard_velocity", standard_velocity,
           "--velocity", velocity, "--raypath",  raypath, "--output", output,
           "--grid_id", str(source[3])]

    return subprocess.Popen(cmd).wait()


def generate(env):
    SensivityAction = \
            Action(Sensivity,
                   strfunction = logger.default_strfun(tool = "Sensivity"))

    CrossSensivityAction = \
            Action(CrossSensivity,
                   strfunction = logger.default_strfun(tool = "CrossSensivity"))
    env['BUILDERS']['Sensivity'] = Builder(action = SensivityAction)
    env['BUILDERS']['CrossSensivity'] = Builder(action = CrossSensivityAction)


def exists(env):
    return 1
