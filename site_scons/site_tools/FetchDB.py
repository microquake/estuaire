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

import agstd.sdb.dbase as sdbase

import string

import logger

import eikonal.data
from eikonal.data import ev_dtype, st_dtype, tt_dtype

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

def init_model_array(dbfile, dbgroup, catname, efilters):
    db = sdbase.SeismicHDF5DB(dbfile, dbgroup)
    ma = sdbase.ModelArray(db, catname)

    for f in efilters:
        fct = getattr(ma, "add_%s_mask" % f['name'])
        fct(*f['args'])

    return ma

def populate_events(ma):
    ary     = np.empty(len(ma.unique_events), dtype = ev_dtype)
    evid    = np.sort(ma.unique_events)
    raw     = ma.events[evid]

    ary['id']       = evid
    ary['delta_t']  = 0
    ary['position'] = np.array([raw[c] for c in ['X', 'Y', 'Z']]).T

    return ary


def populate_stations(ma):
    ary     = np.empty(len(ma.unique_stations), dtype = st_dtype)
    stid    = np.sort(ma.unique_stations)

    ary['delta_t']  = 0
    ary['position'] = np.array([ma.stations[stid][c] for c in ['X', 'Y', 'Z']]).T
    ary['id']       = stid

    return ary


def populate_traveltimes(ma, evary, stary):
    result = [None] * len(ma.unique_stations)
    for i, (st, ev, tr, tt, pick) in enumerate(zip(*ma.split_stations())):
        argsort = np.argsort(ev)

        ary                 = np.empty(len(ev), dtype = tt_dtype)
        ary['id']           = tr[argsort]
        ary['traveltime']   = tt[argsort]
        ary['event_id']     = np.searchsorted(evary['id'], ev[argsort])

        result[i] = dict(ary=ary,
                         station_id=np.searchsorted(stary['id'], st))
    return result


def DBEmitter(target, source, env):
    evfile, stfile = [str(t) for t in target[:2]]

    dbfile = str(source[0])
    dbgroup, catname, efilters, ptypelst = [s.value for s in source[1:5]]

    ftemplate = string.Template(source[5].value)

    ma = init_model_array(dbfile, dbgroup, catname, efilters)

    stargets = []
    for ptype in ptypelst:
        for s in ma.unique_stations:
            stargets.append(ftemplate.substitute(sid = s, ptype = ptype))


    if len(stargets) == 0:
        if logger.tools.isEnabledFor(logger.CRITICAL):
            logger.tools.critical("No traveltime extracted." \
                                  "Could be a bad filter ?")
    return target + stargets, source


def H5FEvent(target, source, env):
    """
    :source 0: dbfile
    :source 1: group
    :source 2: catalog
    :source 3: filters

    :target 0: Event file
    """
    evfile = str(target[0])

    dbfile = str(source[0])
    group, catalog, efilters = [s.value for s in source[1:4]]

    ma = init_model_array(dbfile, group, catalog, efilters)

    evary   = populate_events(ma)
    event_table = eikonal.data.EKEventTable(evary)

    pickle.dump(event_table, open(evfile, 'w'), protocol = pickle.HIGHEST_PROTOCOL)


def H5FStation(target, source, env):
    """
    :source 0: dbfile
    :source 1: group
    :source 2: catalog
    :source 3: filters

    :target 0: Station file
    """
    stfile = str(target[0])

    dbfile = str(source[0])

    group, catalog, efilters = [s.value for s in source[1:4]]

    ma = init_model_array(dbfile, group, catalog, efilters)

    stary   = populate_stations(ma)
    station_table = eikonal.data.EKStationTable(stary)

    pickle.dump(station_table, open(stfile, 'w'), protocol = pickle.HIGHEST_PROTOCOL)


def H5FTT(target, source, env):
    """
    """
    stfile = str(target[0])

    dbfile, stfile, evfile = [str(s) for s in source[:3]]
    group, catalog, efilters, ptype = [s.value for s in source[3:7]]
    ftemplate = string.Template(source[7].value)


    stary = np.load(stfile).data
    evary = np.load(evfile).data

    ma = init_model_array(dbfile, group, catalog, efilters)
    ma.ptype = ptype

    ttarys = populate_traveltimes(ma, evary, stary)
    for tt in ttarys:
        sid = stary['id'][tt['station_id']]
        filename = ftemplate.substitute(sid = sid, ptype = ptype)
        tt_table = eikonal.data.EKTTTable(tt['ary'], tt['station_id'], evnfile = evfile, stafile = stfile)
        pickle.dump(tt_table, open(filename, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL)


def H5FEmitter(target, source, env):
    dbfile, stfile, evfile = [str(s) for s in source[:3]]
    group, catalog, efilters, ptype = [s.value for s in source[3:7]]
    ftemplate = string.Template(source[7].value)

    ma = init_model_array(dbfile, group, catalog, efilters)

    stary   = populate_stations(ma)
    evary   = populate_events(ma)

    ttarys = populate_traveltimes(ma, evary, stary)
    stargets = []
    for tt in ttarys:
        sid = stary['id'][tt['station_id']]
        filename = ftemplate.substitute(sid = sid, ptype = ptype)
        stargets.append(filename)


    if len(stargets) == 0:
        if logger.tools.isEnabledFor(logger.CRITICAL):
            logger.tools.critical("No traveltime extracted." \
                                  " Bad filter(s) ?")

    return target + stargets, source


def FetchDBAction(source, target, env):
    """
    Source :    0   - Database file
                1   - Database group (Value)
                2   - Catalog name (Value)
                3   - Filters
                4   - Output Template

    Target :    0   - Event file
                1   - Station file
                2 ... - TT Files
    """


    import agstd.sdb.dbase as sdbase

    evfile, stfile = [str(t) for t in target[:2]]


    dbfile = str(source[0])
    dbgroup, catname, efilters, ptypelst = [s.value for s in source[1:5]]

    ftemplate = string.Template(source[5].value)

    ma = init_model_array(dbfile, dbgroup, catname, efilters)

    evary   = populate_events(ma)
    stary   = populate_stations(ma)

    station_table = eikonal.data.EKStationTable(stary)
    event_table = eikonal.data.EKEventTable(evary)

    pickle.dump(event_table, open(evfile, 'w'), protocol = pickle.HIGHEST_PROTOCOL)
    pickle.dump(station_table, open(stfile, 'w'), protocol = pickle.HIGHEST_PROTOCOL)
    #np.save(evfile, evary)
    #np.save(stfile, stary)

    for ptype in ptypelst:
        ma.ptype = ptype

        ttarys  = populate_traveltimes(ma, evary, stary)

        for tt in ttarys:
            sid = stary['id'][tt['station_id']]
            filename = ftemplate.substitute(sid = sid, ptype = ptype)
            tt_table = eikonal.data.EKTTTable(tt['ary'], tt['station_id'], evnfile = evfile, stafile = stfile)
            pickle.dump(tt_table, open(filename, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL)


def generate(env):
    H5FEventAction = Action(H5FEvent, strfunction = logger.default_strfun("Fetch Event from H5F DB"))
    H5FStationAction = Action(H5FStation, strfunction = logger.default_strfun("Fetch Station from H5F DB"))

    H5FTraveltimeAction = Action(H5FTT, strfunction = logger.default_strfun("Fetch Traveltime from H5F DB"))
    env['BUILDERS']['FetchDB'] = \
            Builder(action = Action(FetchDBAction,
                                    strfunction = logger.default_strfun("Fetch Database")), 
                    emitter = DBEmitter)

    env['BUILDERS']['H5FFetchEvent'] = Builder(action = H5FEventAction)
    env['BUILDERS']['H5FFetchStation'] = Builder(action = H5FStationAction)
    env['BUILDERS']['H5FFetchTraveltime'] = Builder(action = H5FTraveltimeAction, emitter = H5FEmitter)


def exists(env):
    return 1
