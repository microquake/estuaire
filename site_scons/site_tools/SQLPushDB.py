#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = \
"""
"""

from SCons.Script import Builder, Action
import sqlite3

import pickle
import numpy as np

import agstd.sdb.sqldbase as dbase


def np_load(*args, **kwargs):
    if 'allow_pickle' in kwargs.keys():
        return np.load(*args, **kwargs)
    else:
        return np.load(*args, allow_pickle=True, **kwargs)


def SQLPushDB(target, source, env):
    """
    
    dbfile = str(target[0])
    conn = sqlite3.connect(dbfile)
    conn.executescript(dbase.__DEFAULT_SEISMIC_DB__)
    conn.commit()

    evfile, stfile = [str(s) for s in source[:2]]
    ttfiles = [str(s) for s in source[2:]]

    evtable = np_load(open(evfile))
    sttable = np_load(open(evfile))

    for i, e in enumerate(evtable):
        conn.execute("INSERT INTO event(id, X, Y, Z) VALUES(?,?,?,?)",
                     (i, e['position'][2], e['position'][1], e['position'][0]))

    for i, s in enumerate(sttable):
        conn.execute("INSERT INTO station(id, X, Y, Z) VALUES(?,?,?,?)",
                     (i, s['position'][2], s['position'][1], s['position'][0]))

    i = 0
    for ttfile in ttfiles:
        tttable = np.load(open(ttfile))
        sid = tttable['station_id']
        for tt in tttable['ary']:
            conn.execute("INSERT INTO" \
                         "seismogram(id, event_id, station_id, date, frame, sample_rate)" \
                         "VALUES(?,?,?,?,?,?)", (i, tt['event_id'], sid, "2000-01-01", 0, 0))



    conn.commit()



def generate(env):
    env['BUILDERS']['SQLPushDB'] = Builder(action = SQLPushDB)

def exists(env):
    return 1
