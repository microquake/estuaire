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

import logger

def EuclidianTransform(source, target, env):
    """
    Source :        0 - Input Files
                    1 - Origin (Value)
                    2 - Scale   (Value)

    Target :        0 - Output Files
    """
    origin, scale = [s.value for s in source[1:3]]
    inp = str(source[0])
    out = str(target[0])

    obj = pickle.load(open(inp))
    obj.position = (obj.position - origin) *  scale
    obj.origin += origin
    obj.scale *= scale
    pickle.dump(obj, open(out, 'w'), protocol = pickle.HIGHEST_PROTOCOL)

def generate(env):
    env['BUILDERS']['TableTransform'] = \
            Builder(action = Action(EuclidianTransform,
                                    strfunction = logger.default_strfun("Table Transform")))


def exists(env):
    return 1
