#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#

__doc__ = """
"""

from SCons.Script import *

def BuildInfoAction(target, source, env):
    with open(str(target[0]), 'w') as f:
        lines = ['%s\n\n' % datetime.datetime.now().isoformat(),
                'Builded by %s on\n' % os.getlogin(),
                '%s\n' % ' '.join(os.uname()[:3]),
                ]
        f.writelines(lines)

def exists(env):
    return 1

def generate(env):
    env['BUILDERS']['BuildInfos'] = Builder(action = BuildInfoAction)
