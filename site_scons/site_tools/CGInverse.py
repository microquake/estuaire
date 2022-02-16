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

try:
    import simplejson as json
except ImportError:
    import json

import subprocess

import logger
import escripts


def CGAction(target, source, env):
    """
    Source :        0 - A + R Matrix
                    1 - Prior File
                    2 - Smoothing File
                    3 - Initial Model
                    4 - Weighting scheme (Value)
                    5 - gtol
                    6 - inversion type

    Target :        0 - Result

    """
    A_file, P_file, D_file, I_file = [str(s) for s in source[:4]]
    weighting = json.dumps(source[4].value)

    batch = env['CG_BATCH_SIZE']
    iter = env['CG_MAX_ITER']
    gtol = source[5].value

    cmd = [escripts.CONJUGATE_GRADIENT, "--A", A_file, "--D", D_file,
           "--P", P_file, "--weighting", weighting, "--batch", str(batch),
           "--max_iter", str(iter), "--gtol", str(gtol), "--output",
           str(target[0]), "--initial", I_file, "--stats", str(target[1]),
          "--itype", source[6].value]

    s = subprocess.Popen(cmd)
    rcode = s.wait()

    #if logger.structured.isEnabledFor(logger.logging.DEBUG):
    #    logger.structured.debug(s.stdout.read())

    return rcode


def CGInverse(env, target, source, gtol = None):
    """
    Found Solution to the problem :
        (Ax - b)^2 + (m0 + x)D^2 + (xP)^2

    Source :        0 - List Grid (Value/List)
                    1 - Damping/Priors (Value/List/File)
                    2 - Smoothing/Flatening (Value/List/File)
                    3 - InitialModel (Value/List/File)
                    4 - Weighing Scheme (Value/Dict)
                    5 - gtol
                    4 ... n - Column Files


    Target :        0 - Result File
                    1 - Problem File (A + residual)
                    2 - Residual File
                    3 - Prior File
                    4 - Smothing File
                    5 - Initial Model File
                    6 - Description File
                    7 - Statistic File
    """
    colfiles    = source[5]
    colblocks   = source[0]

    prior, smoothing, imodel = source[1:4]
    weighting = source[4]

    A_file, R_file, P_file, D_file, I_file, desc_file = [str(t) for t in target[1:7]]
    A = env.BuildQuadratic([A_file, R_file, desc_file], [colblocks] + colfiles)

    D = env.BuildBlkMatrix([D_file], [desc_file] + smoothing)
    P = env.BuildBlkMatrix([P_file], [desc_file] + prior)
    I = env.BuildBlkVector([I_file], [desc_file] + imodel)

    gtol = env['CG_GTOL'] if gtol is None else gtol

    env.CGInverseProblem([target[0], target[7]],
                         [A[0], P[0], D[0], I[0], weighting, env.Value(gtol)])
    return [[result[0]], A[0], A[1], P, D, I, A[2]]


def CGInverse2(env, A_file, R_file, P_file, D_file, I_file, desc_file):
    """
    Found Solution to the problem :
        (Ax - b)^2 + (m0 + x)D^2 + (xP)^2

    Source :        0 - List Grid (Value/List)
                    1 - Damping/Priors (Value/List/File)
                    2 - Smoothing/Flatening (Value/List/File)
                    3 - InitialModel (Value/List/File)
                    4 - Weighing Scheme (Value/Dict)
                    5 - gtol
                    4 ... n - Column Files


    Target :        0 - Result File
                    1 - Problem File (A + residual)
                    2 - Residual File
                    3 - Prior File
                    4 - Smothing File
                    5 - Initial Model File
                    6 - Description File
                    7 - Statistic File
    """
    colfiles    = source[5]
    colblocks   = source[0]

    prior, smoothing, imodel = source[1:4]
    weighting = source[4]

    A_file, R_file, P_file, D_file, I_file, desc_file = [str(t) for t in target[1:7]]
    A = env.BuildQuadratic([A_file, R_file, desc_file] , [colblocks] + colfiles)

    D = env.BuildBlkMatrix([D_file], [desc_file] + smoothing)
    P = env.BuildBlkMatrix([P_file], [desc_file] + prior)
    I = env.BuildBlkVector([I_file], [desc_file] + imodel)

    gtol = env['CG_GTOL'] if gtol is None else gtol

    env.CGInverseProblem([target[0], target[7]],
                         [A[0], P[0], D[0], I[0], weighting, env.Value(gtol)])
    return [[result[0]], A[0], A[1], P, D, I, A[2]]


def generate(env):
    env.CGInverse = CGInverse.__get__(env, env.__class__)
    env.CGInverse2 = CGInverse2.__get__(env, env.__class__)

    env['BUILDERS']['CGInverseProblem'] =\
            Builder(action=Action(CGAction,
                                  strfunction=logger.default_strfun(
                                      "Conjugate Gradient")))
    env['CG_BATCH_SIZE'] = 25
    env['CG_MAX_ITER'] = 10000

    env.Tool('BuildQuadratic')


def exists(env):
    return 1
