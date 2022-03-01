#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = """
"""

try:
    import json
except ImportError as e:
    import simplejson as json

import SCons
from SCons.Script import Builder, Action


import scipy as sc
import scipy.sparse

import numpy as np
import pickle

import logger


def np_load(*args, **kwargs):
    if 'allow_pickle' in kwargs.keys():
        return np.load(*args, **kwargs)
    else:
        return np.load(*args, allow_pickle=True, **kwargs)


def CreateBlockMatrix(ffiles, col):
    coldict = dict(zip(col, range(len(col))))

    sparse = []
    residual = []
    description = {}

    for j, f in enumerate(ffiles):
        d = sc.load(str(f))
        current_row = [None] * len(col)
        current_r = d['residual']
        for k in col:
            if k in d:
                current_row[coldict[k]] = d[k]
                if k not in description:
                    description[k] = d['description'][k]
        if any([c != None for c in current_row]):
            sparse.append(current_row)
            residual.append(current_r)


    R = np.concatenate(residual)

    return sparse, R, description


def BuildQuadraticAction(source, target, env):
    """
    Source :        0 - Filter (List/Value)
                    1 ... n - Source files

    Target :        0 - Quadratic Problem Matrix
                    1 - Residual File
                    2 - Problem Description
    """
    filt = source[0].value
    ffiles = source[1:]

    A, R, description = CreateBlockMatrix(ffiles, filt)

    A = sc.sparse.bmat(A)

    A = A.tocsr()





    pickle.dump(dict(A = A, R = R),
                open(str(target[0]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)

    pickle.dump(R,
                open(str(target[1]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)


    pickle.dump([(f, description[f]) for f in filt],
                open(str(target[2]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)


def CreateBlkMatrixAction(source, target, env):
    """
    Source :        0 - Description File
                    1 ... n Value/File


    Target :        0 - Output file
    """
    description = pickle.load(open(str(source[0], 'rb')))
    blk = [[None] * len(description) for i in description]
    sparse = []
    for i, (s, (name, shape)) in enumerate(zip(source[1:], description)):
        size = np.prod(shape)
        row = [None] * len(description)
        if isinstance(s, SCons.Node.Python.Value):
            #print len(sc.sparse.eye(size, size, dtype = 'float'))
            row[i] = sc.sparse.eye(size, size, dtype = 'float') * s.value
            sparse.append(row)
        else:
            cblk = np_load(str(s))
            if cblk is not None:
                row[i] = cblk
                sparse.append(row)
    mat = sc.sparse.bmat(sparse).tocsr()

    pickle.dump(mat, open(str(target[0]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)


def CreateBlkVectorAction(source, target, env):
    """
    Source :        0 - Description File
                    1 ... n Value/File


    Target :        0 - Output file
    """
    description = pickle.load(open(str(source[0], 'rb')))
    blk = [None] * len(description)
    for i, (s, (name, shape)) in enumerate(zip(source[1:], description)):
        size = np.prod(shape)
        if isinstance(s, SCons.Node.Python.Value):

            blk[i] = np.diag([s.value] * size)
        else:
            b = np_load(str(s))
            if b.size != size:
                raise AttributeError("Invalid Vector Size Detected : %s [%d vs %d]" % (name, b.size, size))
            blk[i] = np.ravel(np_load(str(s)))
    vect = np.concatenate(blk)

    pickle.dump(vect, open(str(target[0]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)

def generate(env):
    env['BUILDERS']['BuildBlkMatrix'] = \
            Builder(action = Action(CreateBlkMatrixAction,
                                    strfunction = logger.default_strfun("Block Matrix")))

    env['BUILDERS']['BuildBlkVector'] = \
            Builder(action = Action(CreateBlkVectorAction,
                                    strfunction = logger.default_strfun("Block Vector")))

    env['BUILDERS']['BuildQuadratic'] = \
            Builder(action = Action(BuildQuadraticAction,
                                    strfunction = logger.default_strfun("Quadratic")))


def exists(env):
    return 1
