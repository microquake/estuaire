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

import logger

import agstd.sdb.sqldbase as sqldbase
import sqlite3
import string
import eikonal.data

import pickle

import numpy as np


def split_station(table, ids):
    sorted_value = np.searchsorted(table['station_id'], ids, side = 'right')
    return np.split(table, sorted_value)


def SQLTTEmitter(target, source, env):
    dbfile, evnfile, stafile = [str(s) for s in source[0:3]]
    catalog, efilter = [s.value for s in source[3:5]]

    dbfile = str(source[0])

    conn = sqlite3.connect(dbfile)

    builder = sqldbase.ModelQueryBuilder()
    builder.set_filters(efilter)

    station = conn.execute(builder.station_query(catalog=catalog)).fetchall()
    station = np.array(station, dtype=sqldbase.st_dtype)

    ftemplate = string.Template(source[5].value)

    stargets = [ftemplate.substitute(sid=s) for s in station['id']]

    if len(stargets) == 0:
        if logger.tools.isEnabledFor(logger.CRITICAL):
            logger.tools.critical("Database extraction ended with no "
                                  "traveltime" \
                                  "being selected. Check your filter ... ")

    return stargets, source


def SQLTT(target, source, env):
    """
    """
    ttfiles = [str(t) for t in target]

    dbfile, evnfile, stafile = [str(s) for s in source[0:3]]
    catalog, efilter = [s.value for s in source[3:5]]

    conn = sqlite3.connect(dbfile)

    builder = sqldbase.ModelQueryBuilder()
    builder.set_filters(efilter)

    event = pickle.load(open(evnfile, 'rb'))
    station = pickle.load(open(stafile, 'rb'))
    tt = conn.execute(builder.traveltime_query(catalog=catalog)).fetchall()
    tt = np.array(tt, dtype=sqldbase.tt_dtype)

    tt['event_id'] = np.searchsorted(event.data['id'], tt['event_id'])

    tts = split_station(tt, station.data['id'])
    #print tts
    for sid, (t, out) in enumerate(zip(tts, ttfiles)):
        tt_table = eikonal.data.EKTTTable(t, sid, evnfile = evnfile, stafile = stafile)
        pickle.dump(tt_table, open(out, 'wb'),
                        protocol = pickle.HIGHEST_PROTOCOL)


def SQLStation(target, source, env):
    """
    """
    stfile = str(target[0])

    dbfile = str(source[0])
    catalog, efilter = [s.value for s in source[1:3]]

    conn = sqlite3.connect(dbfile)

    builder = sqldbase.ModelQueryBuilder()
    builder.set_filters(efilter)

    station = conn.execute(builder.station_query(catalog = catalog)).fetchall()

    station = [(s[0], s[1], (s[4], s[3], s[2]), 0) for s in station]

    station = np.array(station, dtype = eikonal.data.st_dtype)

    station_table = eikonal.data.EKStationTable(station)

    pickle.dump(station_table, open(stfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def SQLEvent(target, source, env):
    """
    """
    evfile = str(target[0])

    dbfile = str(source[0])
    catalog, efilter = [s.value for s in source[1:3]]

    conn = sqlite3.connect(dbfile)

    builder = sqldbase.ModelQueryBuilder()
    builder.set_filters(efilter)

    event = conn.execute(builder.event_query(catalog = catalog)).fetchall()

    event = [(e[0], e[1], (e[4], e[3], e[2]), 0) for e in event]

    event = np.array(event, dtype = eikonal.data.ev_dtype)

    event_table = eikonal.data.EKEventTable(event)

    pickle.dump(event_table, open(evfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)


def generate(env):
    SQLEventAction = Action(SQLEvent, strfunction = logger.default_strfun("Fetch Event from SQL DB"))
    SQLStationAction = Action(SQLStation, strfunction = logger.default_strfun("Fetch Station from SQL DB"))

    SQLTraveltimeAction = Action(SQLTT, strfunction = logger.default_strfun("Fetch Traveltime from SQL DB"))

    env['BUILDERS']['SQLFetchStation'] = \
            Builder(action = SQLStationAction)

    env['BUILDERS']['SQLFetchEvent'] = \
            Builder(action = SQLEventAction)

    env['BUILDERS']['SQLFetchTraveltime'] = \
            Builder(action = SQLTraveltimeAction,
                    emitter = SQLTTEmitter)

def exists(env):
    return 1

