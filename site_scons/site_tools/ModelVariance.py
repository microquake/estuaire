#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
import SCons
from SCons.Script import Builder, Action

import pickle

import logger

import numpy as np
import scipy as sc
import scipy.sparse

def NormalizePrior(target, source, env):
    """
        Target :    0 - n result file

        Source :    0 - Problem Description
                    1 - Frechet Matrix
                    2 - Column Name (equal number of result)
                    3 ... n + 3 - Priors
    """
    pdescfile, frechetfile = [str(s) for s in source[:2]]

    desc = pickle.load(open(pdescfile, 'rb'))
    A = pickle.load(open(frechetfile, 'rb'))['A']
    logger.tools.info(str(desc))

    for cname, output, prior in zip(source[2].value, target[:], source[3:]):
        pos = 0
        for dname, shape in desc:
            if dname == cname:
                break
            else:
                pos += np.prod(shape)
        size = np.prod(shape)
        sub_matrix = A[:,pos:pos + size]
        vect = sub_matrix.multiply(sub_matrix).sum() 
        norm = vect / float(size)
        logger.tools.critical(str(norm))
        if isinstance(prior, SCons.Node.Python.Value):
            mat = sc.sparse.eye(size, size) * (norm / prior.value)
            logger.tools.critical(str(prior.value))
        else:
            mat = pickle.load(open(prior, 'rb')) / prior.value
        pickle.dump(mat.tocsr(), open(str(output), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



def generate(env):
    env['BUILDERS']['NormalizePrior'] = Builder(action = NormalizePrior)

def exists(env):
    return 1
