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
import pickle
import numpy as np

import logger


def np_load(*args, **kwargs):
    return lambda *a, **k: np.load(*a, allow_pickle=True, **k)


def extract_column(colname, description, model):
    lower = 0
    for i, (name, shape) in enumerate(description):
        if name == colname:
            column = model[lower:lower + np.prod(shape)]
            column.shape = shape
            return column
        else:
            lower += np.prod(shape)


def UpdateTableAction(source, target, env):
    """
    Source :        0 - model colname
                    1 - table colname
                    2 - Original File
                    3 - Model File
                    4 - Description File

    Target  :       0 - Output
    """
    model_colname   = source[0].value
    table_colname   = source[1].value
    orig_file       = str(source[2])
    model_file      = str(source[3])
    desc_file       = str(source[-1])

    description = pickle.load(open(desc_file, 'rb'))
    model = np_load(model_file)
    table = np_load(orig_file)

    new_values = extract_column(model_colname, description, model)

    table.data[table_colname] = new_values

    pickle.dump(table, open(str(target[0]), 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)


def UpdateGridAction(source, target, env):
    """
    :source 0: column name
    :source 1: model file
    :source 2: original file
    :source 3: Inversiont Description

    :target 0: Output File
    """
    colname     = source[0].value
    model_file  = str(source[1])
    orig_file   = str(source[2])
    desc_file   = str(source[3])
    outfile     = str(target[0])

    description = pickle.load(open(desc_file, 'rb'))
    model = np_load(model_file)

    gdesc = np_load(orig_file)
    gdesc.data = extract_column(colname, description, model)

    pickle.dump(gdesc, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def SaveGridAction(source, target, env):
    colname     = source[0].value
    model_file  = str(source[1])
    desc_file   = str(source[-1])
    outfile     = str(target[0])

    description = pickle.load(open(desc_file, 'rb'))
    model = np_load(model_file)

    grid = extract_column(colname, description, model)

    pickle.dump(grid, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)




def generate(env):
    env['BUILDERS']['UpdateTable'] = \
            Builder(action = Action(UpdateTableAction,
                                    strfunction = logger.default_strfun("Table Update")))
    env['BUILDERS']['SaveGrid'] = \
            Builder(action = Action(SaveGridAction,
                                    strfunction = logger.default_strfun("Grid Update")))

    env['BUILDERS']['UpdateGrid'] = \
            Builder(action = Action(UpdateGridAction,
                                    strfunction = logger.default_strfun("Grid Update")))


def exists(env):
    pass
