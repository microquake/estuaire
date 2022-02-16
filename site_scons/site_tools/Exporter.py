#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#

from SCons.Script import Builder
import subprocess
import escripts

def ExportTXTGrid(target, source, env):
    cmd = [escripts.EXPORT_TXT_GRID, '--grid', str(source[0]), "--output", str(target[0])]
    return subprocess.Popen(cmd).wait()

def exists(env):
    return True

def generate(env):
    env['BUILDERS']['ExportTXTGrid'] = Builder(action = ExportTXTGrid)
