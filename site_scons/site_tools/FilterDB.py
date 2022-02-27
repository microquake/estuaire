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


def np_load(*args, **kwargs):
    return lambda *a, **k: np.load(*a, allow_pickle=True, **k)


def FilterDB(source, target, env):
    """
    Source :        0 - Event File
                    1 - Station File
                    2 - TT File
                    3 - Origin Position
                    4 - Length

    Target :        0 - Output TT File
    """
    etable, stable, ttable = [np_load(str(s)) for s in source[:3]]
    origin, length = [s.value for s in source[3:5]]

    outfile = str(target[0])

    epos = etable['position']
    spos = stable['position']

    ein = True

    for o, l, e, s in zip(origin, length, epos.T, spos.T):
        ein = (e > o) & (e < (o + l)) & ein

    ttable['ary'] = ttable['ary'][ein[ttable['ary']['event_id']]]

    pickle.dump(ttable, open(outfile, 'wb'))

def UpdateAndFilterTT(source, target, env):
    """
    :source 0:  Traveltime File
    :source 1:  Event File
    :source 2:  Station File
    :source 3:  Grid File
    :source 4:  padding in grid spacing unit

    :target 0:  Output TT FIle
    """
    tttable, evtable, sttable, grid = [np_load(str(s)) for s in source[:4]]
    padding = source[4].value * grid.spacing

    outfile = str(target[0])

    epos = evtable.data['position']
    spos = sttable.data[tttable.station_id]['position']

    tmask = True
    import sys
    for o, s, e in zip(grid.origin, grid.data.shape, spos):
        tmask = (e > (o + padding)) & (e < (o + (s - 1) * grid.spacing  - padding)) & tmask

    for o, s, e in zip(grid.origin, grid.data.shape, epos.T):
        tmask = (e > (o + padding)) & (e < (o + (s - 1) * grid.spacing  - padding)) & tmask


    mask = tmask[tttable.data['event_id']]

    tttable.data = tttable.data[mask]

    tttable.__evn_file__ = str(source[1])
    tttable.__sta_file__ = str(source[2])

    pickle.dump(tttable, open(outfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)






def generate(env):
    env['BUILDERS']['FilterDB'] = Builder(action = Action(FilterDB, strfunction = logger.default_strfun("Filtering")))
    env['BUILDERS']['UpdateAndFilterTT'] = Builder(action = Action(UpdateAndFilterTT, strfunction = logger.default_strfun("Filtering TT")))

def exists(env):
    return 1
