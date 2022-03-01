from SCons.Script import Builder, Action
import numpy as np

import pickle

import logger


def np_load(*args, **kwargs):
    if 'allow_pickle' in kwargs.keys():
        return np.load(*args, **kwargs)
    else:
        return np.load(*args, allow_pickle=True, **kwargs)


def AdditiveGaussianNoise(target, source, env):
    tableobj = np_load(str(source[0]))
    columns, stdev = [s.value for s in source[1:3]]

    for c, s in zip(columns, stdev):
        shape = tableobj.data[c].shape
        tableobj.data[c] += np.random.normal(loc=0, scale=s, size=shape)

    pickle.dump(tableobj, open(str(target[0]), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


def generate(env):
    AdditiveGaussianNoiseAction = Action(AdditiveGaussianNoise,
        strfunction=logger.default_strfun("Additive Gaussian Noise"))

    env['BUILDERS']['AdditiveGaussianNoise'] = \
        Builder(action=AdditiveGaussianNoiseAction)


def exists(env):
    return True
