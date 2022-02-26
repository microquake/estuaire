from SCons.Script import Builder, Action
import numpy as np

import pickle

import logger

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

def AdditiveGaussianNoise(target, source, env):
    tableobj = np.load(str(source[0]))
    columns, stdev = [s.value for s in source[1:3]]

    for c, s in zip(columns, stdev):
        shape = tableobj.data[c].shape
        tableobj.data[c] += np.random.normal(loc=0, scale=s, size=shape)

    pickle.dump(tableobj, open(str(target[0]), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def generate(env):
    AdditiveGaussianNoiseAction = Action(AdditiveGaussianNoise,
                                         strfunction = logger.default_strfun("Additive Gaussian Noise"))
    env['BUILDERS']['AdditiveGaussianNoise'] = Builder(action = AdditiveGaussianNoiseAction)

def exists(env):
    return True
