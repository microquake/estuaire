#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#


__doc__ = """
Lotic is actually a collection of many small SCons tools which helps
interfacing with the eikonal executable scripts.
"""

__regularization_targets__ = {}

import string
import os


def Raytracing(env, eikonal_targets,
               traveltime_targets, velocity_grids, root = 'build', h = 1):
    """
    """
    rtemplate = string.Template(env['FORWARD_RAY_TEMPLATE'])
    rays_targets = []
    for grid, ttfiles, efiles in zip(velocity_grids, traveltime_targets, eikonal_targets):
        rtargets = []
        for tt, etgt in zip(ttfiles, efiles):
            basename = os.path.basename(str(tt))
            rbasename = rtemplate.substitute(basename = basename)
            routput = os.path.join(root, env['FORWARD_ROOT'], rbasename)
            rtargets.append(env.Raytrace([routput],
                                         [grid, etgt, tt, env.Value(h)])[0])
        rays_targets.append(rtargets)
    return rays_targets
Raytracing.__description__ = "Raytracing"


def RegularizationOperator(env, shape, eta = (1, 1), spacing = 1):
    """
    This method create the appropriate linear smoothing operator for the
    given parameters and ensure. The resulting operator target is
    <Memoized> and each call to this function with the same set of
    parameters will always yield the same target. The directory where
    the target are produced is environment tied and can be modified by the
    user.
    """
    template = string.Template(env['GRID_REGULARIZATION_TEMPLATE'])
    output_filename = template.substitute(shape = "x".join(map(str,shape)),
                                          eta = "_".join(map(str, eta)))


    if output_filename not in __regularization_targets__:
        source = map(env.Value, [shape, eta, spacing])
        __regularization_targets__[output_filename] =\
                env.GridDifference([output_filename],[source])

    return __regularization_targets__[output_filename]


def generate(env):

    tools = ['BuildInfos', 'ForwardModelling', 'FetchDB', 'ExtractColumn',
             'TableTransform', 'VelocityGrid', 'CGInverse','FilterDB',
             'GridDifference', 'UpdateModel', 'Report', 'DoubleDifference',
             'GaussianSmoothing', 'Raytrace', 'SQLFetchDB', 'Checkerboard',
             'Exporter', 'Noise']

    for t in tools:
        env.Tool(t)

    env.RegularizationOperator = \
            RegularizationOperator.__get__(env, env.__class__)

    env.Raytracing = Raytracing.__get__(env, env.__class__)

    env['FORWARD_ROOT'] = 'forward'

    env['FORWARD_EIKONAL_TEMPLATE'] = 'eikonal_${basename}'
    env['FORWARD_FRECHET_TEMPLATE'] = 'frechet_${basename}'
    env['FORWARD_RAY_TEMPLATE']     = 'rays_${basename}'
    env['FORWARD_DD_TEMPLATE']      = 'dd_${basename}'
    env['FORWARD_EP_TEMPLATE']      = 'ep_${basename}'
    env['FORWARD_ST_TEMPLATE']      = 'st_${basename}'
    env['FORWARD_ET_TEMPLATE']      = 'et_${basename}'
    env['INVERSION_ROOT']           = 'inverse'
    env['FORWARD_ROOT']             = 'forward'
    env['DD_ROOT']                  = 'double_difference'
    env['EP_ROOT']                  = 'repositioning'
    env['ST_ROOT']                  = 'stations_time'
    env['ET_ROOT']                  = 'events_time'

    env['INVERSION_DD_EVENTS_COLUMN_VECTOR'] = 'dd_events_vector.npy'
    env['INVERSION_EP_EVENTS_COLUMN_VECTOR'] = 'ep_events_vector.npy'
    env['INVERSION_DD_RESULT_EVENTS']        = 'dd_events.npy'

    env['INVERSION_ST_STATIONS_COLUMN_VECTOR'] = 'st_stations_vector.npy'
    env['INVERSION_ET_EVENTS_COLUMN_VECTOR'] = 'et_events_vector.npy'

    env['INVERSION_DD_EVENTS_COLUMN_VECTOR'] = 'ep_events_vector.npy'
    env['INVERSION_EP_RESULT_EVENTS']        = 'ep_events.npy'


    env['INVERSION_ST_RESULT']     = 'st_stations.npy'
    env['INVERSION_ET_RESULT']     = 'et_events.npy'

    env['INVERSION_PRIOR_MATRIX']       = 'prior_matrix.npy'
    env['INVERSION_SMOOTHING_MATRIX']   = 'smoothing_matrix.npy'
    env['INVERSION_INITIAL_VECTOR']     = 'initial_vector.npy'
    env['INVERSION_DESCRIPTION']        = 'problem_description.npy'
    env['INVERSION_RESIDUAL_VECTOR']    = 'residual_vector.npy'
    env['INVERSION_FRECHET_MATRIX']     = 'frechet_matrix.npy'
    env['INVERSION_RESULT_VECTOR']      = 'result_vector.npy'
    env['INVERSION_STATS_FILE']         = 'stats.npy'

    env['INVERSION_TOMOGRAPHY_GTOL']    = 1e-20
    env['INVERSION_DD_GTOL']            = 1e-6
    env['INVERSION_EP_GTOL']            = 1e-20
    env['INVERSION_ST_GTOL']            = 1e-20
    env['INVERSION_ET_GTOL']            = 1e-20

    env['INVERSION_LOG_FILE']           = "test.a"

    regularization_template = os.path.join('build', 'grid_smoothing',
                                           'regularization-${shape}-${eta}.lop')

    env['GRID_REGULARIZATION_TEMPLATE'] = regularization_template

def exists(env):
    return 1
