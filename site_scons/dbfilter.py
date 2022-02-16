#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
#
__doc__ = """
This module facilitate the use of filter in the FetchDB application.
Since the filtering process used json description, those objects will
ensure the filters are well built Remote Procedure Call will behave
correctly.
"""


def CuboidFilter(origin, length, padding = 0.0):
    """
    This filter remove events which are locate outside of the cube defined
    by the origin and the given length. The optional argument padding can
    also be provided and add a certain percentage of the cube being also
    filtered.
    """
    if len(origin) != len(length):
        raise ValueError("Origin and Length must be array of the same size")
    return dict(name = "cuboid", args = [tuple(origin), tuple(length), padding])


def DateFilter(fromdate, todate):
    """
    This filter remove the events which happened before the fromdate argument
    and after the todate argument. Both date can be provided as a string
    parsable by dateutil parser (e.g. datetime object isoformat) or as the
    datetime object.
    """
    if not isinstance(fromdate, str):
        fromdate = fromdate.isoformat()
    if not isinstance(todate, str):
        todate = todate.isoformat()
    return dict(name = "date", args = [fromdate, todate])

def StationFilter(stations):
    """
    This filter remove the station(s) associated with the given station id(s).
    The parameter can be either a single station id or a list of station ids.
    .. WARNING : Not Implemented in SQL DB
    """
    return dict(name = "station", args = [stations])

def EventFilter(events):
    """
    This filter remove the events associated with the given event id(s).
    The parameter can be either a single event id or a list of event ids.

    .. WARNING : Not implemented in SQL DB
    """
    return dict(name = "event", args = [events])

def PickTypeFilter(ptype):
    """
    This filter only work on traveltime
    """
    return dict(name = "ptype", args = [ptype])

