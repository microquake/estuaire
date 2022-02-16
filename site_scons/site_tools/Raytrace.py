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

import escripts
import logger

def Raytrace(source, target, env):
    """
    :source 0: Velocity Grid
    :source 1: Arrival Grid
    :source 2: TT File
    :source 3: Raytracing Step (Optional) 

    :target 0: Output file
    """

    velocity, arrival, ttfile = [str(s) for s in source[:3]]

    output = str(target[0])

    cmd = [escripts.RAYTRACE, "--arrival", arrival, "--velocity", velocity,
           "--traveltime", ttfile, "--output", output]

    if len(source) > 3:
        cmd.extend(['--h', str(source[3].value)])

    return subprocess.Popen(cmd).wait()

def generate(env):
    RaytraceAction = Action(Raytrace, strfunction = logger.default_strfun("Raytracing"))
    env['BUILDERS']['Raytrace'] = Builder(action = RaytraceAction)


def exists(env):
    return 1
