#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.

import subprocess
import escripts

def DoubleDifference(source, target, env):
    """
    Source :        0 - Input file

    Target :        0 - Output file
    """
    cmd = [escripts.DOUBLE_DIFFERENCE, "--input_file", str(source[0]),
           "--output", str(target[0])]

    return subprocess.Popen(cmd).wait()



from SCons.Script import Builder, Action
import logger
def generate(env):
    env['BUILDERS']['DoubleDifference'] = \
            Builder(action = Action(DoubleDifference,
                                    strfunction = logger.default_strfun("Double Difference")))



def exists(env):
    return 1
