#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = """
:platform: Unix

Slopes is a high level API aim for object-oriented multiple inversion. \
        If you only learn one API it should probably be the easiest and almost\
        everything is accessible within this API. The main idea is to provide\
        an easily manageable objects instead of low level SCons targets since\
        the number of target can grow very fast. The main interfaces is the \
        InversionPlan which is basically an inversion tree controlling the \
        inversion flow.

.. moduleauthor:: Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>

API Description
------------------

.. autoclass:: InversionPlan2
    :members: __init__, insert_velocity_grid, insert_velocity_grid_desc, \
            set_grid_inversion, set_double_difference, set_event_time_correction, \
            set_station_time_correction, set_event_position_correction, push_inversion

.. autoclass:: SQLDBFetchPlan
    :members: set_cuboid_filter, set_date_filter, fetch_traveltime,\
            get_transformed_station, get_transformed_event, set_grid_filter

.. autoclass:: GridDescription
    :members: fromlimits, transform_to_grid_coordinates

"""

try:
    import json
except ImportError as e:
    import simplejson as json

import os
import SCons.Script
import SCons.Node
import weakref

from agstd.decorators import memoize, deprecated

import copy

import eikonal.data

import numpy as np
import logger

import string

import dbfilter

import logging
log = logging.getLogger("tools.Slopes")

from SCons.Script import Action

import pickle


def exprange(p, max_value, min_value, size=30):
    this_range = max_value - min_value
    return np.exp(np.linspace(p, 0, size)) * this_range / np.exp(p) + min_value


def __uid_generator__():
    i = 0
    while i < 0xFFFFFFFF:
        yield i
        i += 1


def UniformWeighting():
    """
    """
    return dict(name = "uniform", args = [])

def GaussianWeighting(sigma):
    """
    """
    return dict(name = "gaussian", args = [sigma])

def BoxWeighting(sigma):
    """
    """
    return dict(name = "box", args = [sigma])

def InfoFct(target, source, env):
        rfile = str(source[0])
        node_dict = source[1].value
        R = pickle.load(open(str(source[0])))

        if logger.structured.isEnabledFor(logger.INFO):
            avg = np.average(R)
            std = np.std(R)
            rms = np.average(np.sqrt(R ** 2))

        log_description = dict(residual = dict(average = avg, stdev = std, rms = rms),
                               ntraveltime = R.size)
        log_description.update(node_dict)
        logger.structured.info(json.dumps(log_description))

InfoAction = Action(InfoFct, strfunction = lambda x, y, z : "")

class InversionDescription(object):
    def __init__(self):
        self.grids = []
        self.double_difference  = False
        self.event_time         = False
        self.station_time       = False
        self.event_position     = False

class TreeNode(object):
    """
    Simple representation of a node in Tree
    """
    inherited_attributes = ['events', 'stations']
    inherited_collections = ['vgrids']

    __uid_generator__ = __uid_generator__()
    def __init__(self, parent = None, **kw):
        self.uid =self.__uid_generator__.next()
        self.__dict__.update(kw)
        self.childs = []
        self.tag = None
        self.idescription = dict(grids = [], double_difference = False,
                                 event_time = False, station_time = False,
                                 event_position = False)
        self.__parent__ = weakref.ref(parent) if parent is not None else None

    def __get_parent__(self):
        return None if self.__parent__ is None else self.__parent__()
    parent = property(__get_parent__)

    def add_new_child(self):
        kw = {}
        for attr in self.inherited_attributes:
            kw[attr] = getattr(self, attr)
        for attr in self.inherited_collections:
            kw[attr] = copy.deepcopy(getattr(self, attr))
        node = TreeNode(parent = self, **kw)
        self.childs.append(node)
        return node

def __get_list_i__(i):
    def get(self):
        return self[i]
    def set(self, val):
        self[i] = val
    return get,set

class VelocityContainer(list):
    def __init__(self, *args):
        super(VelocityContainer, self).__init__(args)

    def __deepcopy__(self, name):
        return VelocityContainer(*self)

    tgt             = property(*__get_list_i__(0))
    eikonal_tgts    = property(*__get_list_i__(1))
    frechet_tgts    = property(*__get_list_i__(2))
    ttime_tgts      = property(*__get_list_i__(3))
    ray_tgts        = property(*__get_list_i__(4))

frenames  = ['event_time', 'station_time', 'event_position']
nodeattr  = ['events', 'stations', 'events']
colnames  = ['delta_t', 'delta_t', 'position']

class InversionResult(object):
    def __init__(self, env, result, stats, A, R, pdescription, D, P, I,
                 idescription, iroot):
        self.result_vector          = result
        self.stats                  = stats
        self.frechet_matrix         = A
        self.residual_vector        = R
        self.pdescription           = pdescription
        self.idescription           = idescription
        self.regularization_matrix  = D
        self.prior_matrix           = P
        self.initial_vector         = I
        self.iroot                  = iroot
        self.env                    = env

    def totuple(self):
        return ((self.result_vector, self.stats),
                (self.frechet_matrix, self.residual_vector, self.pdescription),
                (self.regularization_matrix, self.prior_matrix,
                 self.initial_vector))

    def update_tables(self, node):
        """
        :param node: The node where we update the tables
        # This Section retrieve the table from the result vector and the
        # problem description
        """
        idescription = self.idescription
        env = self.env

        for c, inattr, col in zip(frenames, nodeattr, colnames):
            if not isinstance(idescription[c], bool) or idescription[c]:
                infile = getattr(node, inattr)
                outfile = os.path.join(self.iroot, "%s_optimized.pickle" % c)
                tgt = env.UpdateTable([outfile],
                                      [env.Value(c), env.Value(col), infile,
                                       self.result_vector, self.pdescription])[0]
                setattr(node, inattr, tgt)

        # Traveltime Filtering
        # Not optimal i know, the traveltime files are always copied to ensure
        # they point to the right event and station file.

        for gid in node.vgrids:
            vgrid = node.vgrids[gid]
            next_tt_tgt = []
            for tt in vgrid.ttime_tgts:
                basename = os.path.basename(str(tt))
                filename = os.path.join(self.iroot, env['INVERSION_ROOT'],basename)

                tgt = env.UpdateAndFilterTT([filename],
                                            [tt, node.events, node.stations, vgrid.tgt, env.Value(1)])[0]
                next_tt_tgt.append(tgt)

            vgrid.ttime_tgts = next_tt_tgt

    def update_velocity(self, node):
        """
        :param node: The node where we update the tables
        Velocity Grid Extraction and Eikonal Solver For Grids which takes
        part in the inversion process.
        """
        env = self.env
        dirname = os.path.join(self.iroot, env['INVERSION_ROOT'])
        etemplate = string.Template(env['FORWARD_EIKONAL_TEMPLATE'])

        for gid, prior, smoothing, post_smoothing in self.idescription['grids']:
            vgrid = node.vgrids[gid]

            gname = gid + ".npy"
            output = os.path.join(self.iroot, gname)
            goutput = output if post_smoothing is None else \
                os.path.join(dirname, "unsmoothed_" + gname)

            next_grid = env.UpdateGrid([goutput], [env.Value(gid),
                                                   self.result_vector,
                                                   vgrid.tgt,
                                                   self.pdescription])[0]
            if post_smoothing is not None:
                next_grid = env.GaussianSmoothing([output],
                                                  [next_grid,
                                                   env.Value(
                                                       post_smoothing)])[0]

            next_eik_tgt = []
            for tt in vgrid.ttime_tgts:
                basename = os.path.basename(str(tt))
                ebasename = etemplate.substitute(basename = basename)

                eoutput = os.path.join(self.iroot, env['FORWARD_ROOT'],
                                       ebasename)

                etgt = env.EikonalSolver2([eoutput], [next_grid, tt])[0]
                env.Depends(etgt, [node.events, node.stations])

                next_eik_tgt.append(etgt)


            vgrid.tgt = next_grid
            vgrid.eikonal_tgts = next_eik_tgt

    def update_frechet(self, node):
        env = self.env

        ftemplate = string.Template(env['FORWARD_FRECHET_TEMPLATE'])
        rtemplate = string.Template(env['FORWARD_RAY_TEMPLATE'])

        for gid in node.vgrids:
            vgrid = node.vgrids[gid]

            next_fre_tgt = []
            next_ray_tgt = []
            for tt, etgt in zip(vgrid.ttime_tgts, vgrid.eikonal_tgts):
                basename = os.path.basename(str(tt))
                fbasename = ftemplate.substitute(basename = basename)
                rbasename = rtemplate.substitute(basename = basename)

                foutput = os.path.join(self.iroot, env['FORWARD_ROOT'],
                                       fbasename)
                routput = os.path.join(self.iroot, env['FORWARD_ROOT'],
                                       rbasename)

                ftgt = env.Sensivity([foutput],
                                     [vgrid.tgt, etgt, tt, env.Value(gid)])[0]
                # We Have to do that to ensure the changed events position are
                # take into account
                env.Depends(ftgt, [node.events, node.stations])
                rtgt = env.Raytrace([routput], [vgrid.tgt, etgt, tt])[0]

                # Needed for god sake
                env.Depends(rtgt, [node.events, node.stations])


                next_fre_tgt.append(ftgt)
                next_ray_tgt.append(rtgt)
            vgrid.ray_tgts = next_ray_tgt
            vgrid.frechet_tgts = next_fre_tgt


class QuadraticBuilderBase(object):
    def __init__(self, env, iroot):
        self.prior                     = []
        self.regularization             = []
        self.columns                    = []
        self.initial                    = []
        self.sensivity                  = []

        self.env                        = env
        self.iroot                      = iroot

    def inverse(self, weighting, gtol = 1e-10, prior_normalization = False,
                itype = 'linearcg'):
        """
        This is the actual building of the Quadratic Problem (Ax - r)^2 = 0.
        This also produce a dictionary of description describing the real
        shape of every part of the problem vector for post interpretation
        purposes.
        """
        env = self.env

        dirname     = os.path.join(self.iroot, self.env['INVERSION_ROOT'])

        A_file      = os.path.join(dirname, env['INVERSION_FRECHET_MATRIX'])
        R_file      = os.path.join(dirname, env['INVERSION_RESIDUAL_VECTOR'])
        desc_file   = os.path.join(dirname, env['INVERSION_DESCRIPTION'])

        D_file      = os.path.join(dirname, env['INVERSION_SMOOTHING_MATRIX'])
        P_file      = os.path.join(dirname, env['INVERSION_PRIOR_MATRIX'])
        I_file      = os.path.join(dirname, env['INVERSION_INITIAL_VECTOR'])

        result_file = os.path.join(dirname, env['INVERSION_RESULT_VECTOR'])
        stats_file  = os.path.join(dirname, env['INVERSION_STATS_FILE'])

        normed_P_file = os.path.join(dirname, 'normed_P.pickle')

        A, R, description = env.BuildQuadratic([A_file, R_file, desc_file],
                                            [env.Value(self.columns)] +
                                               self.sensivity)

        if prior_normalization:
            priors = env.NormalizePrior([normed_P_file],
                                        [description, A,
                                         env.Value(self.columns)] + self.prior)
        else:
            priors = self.prior


        D = env.BuildBlkMatrix([D_file], [desc_file] + self.regularization)[0]
        P = env.BuildBlkMatrix([P_file], [desc_file] + priors)[0]
        I = env.BuildBlkVector([I_file], [desc_file] + self.initial)[0]

        result, istats = self.env.CGInverseProblem([result_file, stats_file],
                                                   [A, P, D, I,
                                                    env.Value(weighting),
                                                    env.Value(gtol),
                                                    env.Value(itype)])

        return InversionResult(env, result, istats, A, R, description, D, P, I,
                               self.idescription, self.iroot)


class QuadraticBuilder(QuadraticBuilderBase):
    """
    """

    def __init__(self, env, iroot, idescription, node):
        QuadraticBuilderBase.__init__(self, env, iroot)
        self.idescription               = idescription
        self.node                       = node

    def process_ev_sta_inversion(self):
        """
        This Section Verify if we run the inversion on any parameter that
        modify station or event file. The first task is to extract the given
        column name from the source file and adding it to the initial value
        vector for the optimization.
        """
        for c, nattr, col in zip(frenames, nodeattr, colnames):
            infile = getattr(self.node, nattr)
            if not isinstance(self.idescription[c], bool) or\
                    self.idescription[c]:
                prior = self.idescription[c]
                column_output = os.path.join(self.iroot, c + ".npy")
                column_vect = self.env.ExtractColumn([column_output],
                                                [infile,
                                                 self.env.Value(col)])[0]

                self.columns.append(c)
                self.initial.append(column_vect)
                self.regularization.append(self.env.Value(0))
                self.prior.append(self.env.Value(prior))

    def process_velocity_inversion(self):
        """
        This Section insert every velocity grid we should run the inversion
        on. Append the grid to the initial vector and add the regularization
        parameter. Since the post-regularization is way more efficient and
        is way faster, currently the smoothing parameter is ignored and
        always filled with zeros.
        """
        for gid, prior, smoothing, post_smoothing in \
                self.idescription['grids']:
            grid = self.node.vgrids[gid].tgt
            naked_gridfile = os.path.join(self.iroot, "%s_data.npy" % gid)
            naked_grid = self.env.ExtractData([naked_gridfile], [grid])
            self.prior.append(self.env.Value(prior))
            self.regularization.append(self.env.Value(0))
            self.initial.append(naked_grid)
            self.columns.append(gid)

    def process_double_difference_inversion(self):
        dtemplate = string.Template("dd_${basename}")
        for gid in self.node.vgrids:
            vgrid = self.node.vgrids[gid]
            fre_tgt = vgrid.frechet_tgts
            if self.idescription['double_difference']:
                new_fre_tgt = []
                for f, tt in zip(vgrid.frechet_tgts, vgrid.ttime_tgts):
                    basename = os.path.basename(str(tt))
                    dbasename = dtemplate.substitute(basename = basename)
                    doutput = os.path.join(self.iroot,
                                           self.env['FORWARD_ROOT'], dbasename)
                    new_fre_tgt.append(self.env.DoubleDifference(
                        [doutput], [f]))
                fre_tgt = new_fre_tgt
            self.sensivity.extend(fre_tgt)



class QATomoQuadraticBuilder(QuadraticBuilderBase):
    """
    """
    def __init__(self, env, iroot, grid, checkerboard, vgrid, gid, sigma):
        QuadraticBuilderBase.__init__(self, env, iroot)

        self.grid         = grid
        self.checkerboard = checkerboard
        self.vgrid        = vgrid
        self.gid          = gid
        self.idescription = dict(grids = [(gid, 0, 0, sigma)])

    def process_cross_velocity_inversion(self):
        """
        """
        env = self.env
        cross_fre = []
        for ray, tt in zip(self.vgrid.ray_tgts, self.vgrid.frechet_tgts):
            basename = os.path.basename(str(tt))
            cs_file = os.path.join(self.iroot, env['FORWARD_ROOT'], basename)

            cs_tgt = env.CrossSensivity([cs_file],
                                        [self.grid, self.checkerboard,
                                         ray, env.Value(self.gid)])
            cross_fre.append(cs_tgt)


        naked_gridfile = os.path.join(self.iroot, "%s_data.npy" % self.gid)
        naked_grid = env.ExtractData([naked_gridfile], [self.grid])

        self.columns.append(self.gid)
        self.sensivity.append(cross_fre)
        self.prior.append(env.Value(0))
        self.regularization.append(env.Value(0))
        self.initial.append(naked_grid)



class InversionPlan(object):
    """
    This object keeps track of the complete inversion procedure and is in \
    charge of building the corresponding SCons targets. The target are pushed \
    onto a target stack and are easily accessible.

    :param env: Scons environment where target will be built
    :param ev_tgt: The target or the filename for the event table
    :param st_tgt: The target or the filename for the station table

    :returns: InversionPlan object
    """
    iroots = []
    def __init__(self, env, ev_tgt, st_tgt):
        self.current            = TreeNode(events = ev_tgt, stations = st_tgt,
                                           vgrids = {})

        self.tags               = {}

        self.report_targets     = []
        self.vgrids             = [{}]

        self.env                = env


        plan_action = Action(self.__plan_infos__,
                             strfunction=lambda x, y, z : "")
        info_tgt = self.env.Command([env.Value("")], [], plan_action)
        env.AlwaysBuild(info_tgt)
        env.Default(info_tgt)

        # Ensure Environment Variable are inside the Environment
        if 'Lotic' not in env['TOOLS']:
            env.Tool('Lotic')

        # Adding a pointer to the root node
        self.tag('root')


    @classmethod
    def from_fetch_plan(cls, fetch_plan):
        return cls(fetch_plan.env, fetch_plan.evnfile, fetch_plan.stafile)


    @staticmethod
    def __plan_infos__(target, source, env):
        if logger.structured.isEnabledFor(logger.INFO):
            logger.structured.info("")
            logger.structured.info("Inversion Plan Execution")
            logger.structured.info("=" * 10)

    def __raytrace__(self, grid, etgt, tt, root):
        basename = "ray_%s" %os.path.basename(str(tt))
        routput = os.path.join(root, self.env['FORWARD_ROOT'], basename)
        rtgt = self.env.Raytrace([routput], [grid, etgt, tt])[0]
        self.env.Depends(rtgt, [self.current.events, self.current.stations])
        return rtgt


    def insert_velocity_grid(self, gid, grid, tt_tgt, root='build', **kw):
        """
        Insert a Velocity grid into the Inversion Plan using the grid
         description
        for the grid parameters

        :param gid: Grid Identifier
        :param gdesc: Grid Description Object
        :param value: Value of the Homogenous Grid
        :param tt_tgt: A list of traveltime associated with the velocity grid
        """
        if gid in self.current.vgrids:
            raise ValueError("Grid named %s already exists" % str(gid))

        filtered_tt = []
        for tt in tt_tgt:
            basename = os.path.basename(str(tt))
            filename = os.path.join(root, gid, basename)
            tgt = self.env.UpdateAndFilterTT([filename],
                                        [tt, self.current.events,
                                         self.current.stations, grid,
                                         self.env.Value(1)])[0]
            filtered_tt.append(tgt)
        tt_tgt = filtered_tt

        eik_tgt, fre_tgt = self.env.ForwardModelling2(tt_tgt, grid, gid,
                                                      os.path.join(root, gid,
                                                      "eikonal_${basename}"),
                                                      os.path.join(root, gid,
                                                      "frechet_${basename}"),)

        rayroot = os.path.join(root, gid)

        ray_tgt = [self.__raytrace__(grid, e, tt, root)
                   for e, tt in zip(eik_tgt, tt_tgt)]

        self.current.vgrids[gid] = VelocityContainer(grid, eik_tgt, fre_tgt,
                                                     tt_tgt, ray_tgt)

        self.__push_report__(gid)

    insert_velocity_grid_desc = insert_velocity_grid

    def set_grid_inversion(self, gid, prior==1, regularization=0,
                           post_smoothing=None):
        """
        Add/Override the velocity into the next inversion iteration with the
        given prior, regularization and post smoothing.

        The velocity grid must already be inserted in the inversion plan.

        :param gid: Grid Identifier
        :param prior: The inversion Prior for the velocity grid
        :param regularization: The weight of the smoothing in the inversion \
                process
        :param post_smoothing: The sigma parameter of the gaussian velocity \
                grid post smoothing

        :returns: None
        """
        self.current.idescription['grids'].append((gid, prior, regularization,
                                                   post_smoothing))

    def set_double_difference(self, dd=True):
        """
        This method controls the inversion double difference operation. If the
        parameter is True, the next inversion matrix will be calculated using
        the double difference linear operator.

        The default value of the dd after with a new state is False

        :param dd: Set or Unset the double difference flag

        :returns: None

        .. note :: Since the double difference operation do not explicitly \
                filter for events nearby, You should probably at least inverse \
                for the velocity grid as well as the event position. This \
                ensure distants events will be ponderate by the tomography \
                between them and more precisely inversed.
        """
        self.current.idescription['double_difference'] = dd

    def __get_double_difference__(self):
        self.current.idescription['double_difference']
    double_difference = property(__get_double_difference__,
                                 set_double_difference)

    def set_event_time(self, prior):
        """
        Add/Override current inversion prior for the event time.

        The prior consist of a diagonal matrix added to the quadratic term
        of the inversion process

        :param prior: The inversion Prior for the event time

        :returns: None
        """

        if self.current.idescription['station_time']:
            raise ValueError("Event Time and Station Time Should Never be "
                             "Inversed toghether")
        self.current.idescription['event_time'] = prior

    set_event_time_correction = set_event_time

    def set_station_time(self, prior):
        """
        Add/Override curremtn inversion prior for the station stime

        The prior consist of a diagonal matrix added to the quadratic term
        of the inversion process

        :param prior: The inversion Prior for the station time

        :returns: None
        """
        if self.current.idescription['event_time']:
            raise ValueError("Event Time and Station Time Should Never be "
                             "Inversed toghether")
        self.current.idescription['station_time'] = prior

    set_station_time_correction = set_station_time

    def set_event_position(self, prior):
        """
        Add/Override current inversion prior for the events position.

        The prior consist of a diagonal matrix added to the quadratic term
        of the inversion process

        :param prior: The inversion Prior for the event position

        :returns: None
        """
        self.current.idescription['event_position'] = prior

    set_event_position_correction = set_event_position

    def tag(self, tag):
        """
        This method tag the current state of the Plan for quick restoration.
        It can be used for example for parameter exploration.

        :param tag: id for the current state (any hashable)

        :returns: None
        """
        self.tags[tag] = self.current
        self.current.tag = tag

    def restore(self, tag):
        """
        Restore the current state to the state pointed by the tag.

        :param tag: id of the state to restore

        :returns: None
        """
        self.current = self.tags[tag]

    def __log_inversion__(self, iroot, idescription, R):
        infodict = dict(uid = str(self.current.uid), description = idescription,
                        rootdir = os.path.abspath(iroot))

        infodict['parent'] = None if self.current.parent is None else str(self.current.parent.uid)

        if self.current.tag is not None:
            infodict['tag'] = self.current.tag
        rval = self.env.Command([self.env.Value(None)],[R, self.env.Value(infodict)],
                           InfoAction)
        self.env.AlwaysBuild(rval)
        self.env.SideEffect('estuaire.log', rval)
        #self.env.Default(rval)

    def __push_report__(self, gid):
        cr = []
        fre_tgt = self.current.vgrids[gid].frechet_tgts
        rreport = self.env.ResidualHistogramReport([], fre_tgt)
        cr.append(rreport[0])
        if len(self.report_targets) != 0:
            self.env.Depends(self.report_targets[-1], rreport)

        self.report_targets.extend(cr)

    def push_inversion(self, iroot, gtol = 1e-10, weighting=UniformWeighting(),
                       report=False, prior_normalization=False,
                       itype='linearcg'):
        """
        This function push the current inversion description to the current
        inversion stack. It build the targets.

        After the inversion, the inversion description is cleaned.

        :param iroot: A base directory where inversion targets will be built
        :param gtol: A lower bound for the gradient magnitude of the inversion.\
                this parameter control when we consider the inversion has \
                converged
        :param weighting: The weighting for the robust residual scheme used \
                for downweighting outliers
        :param report: Produce report target (True/False)
        :param itype: Inversion type (newtoncg/linearcg)

        :returns: None

        .note:: This is where the magic happens
        """
        if iroot in self.iroots:
            if log.isEnabledFor(logging.CRITICAL):
                log.critical("Inversion directory collision detected")
            raise AttributeError("%s is already taken" % iroot)
        else:
            self.iroots.append(iroot)
        if len(self.current.vgrids) == 0:
            raise ValueError("Must at leas have 1 velocity grid")

        env = self.env
        idescription = self.current.idescription

        qbuilder    = QuadraticBuilder(env, iroot, idescription, self.current)

        # Inversion Preparation
        qbuilder.process_ev_sta_inversion()
        qbuilder.process_velocity_inversion()
        qbuilder.process_double_difference_inversion()

        # The actual inversion
        iresult = qbuilder.inverse(weighting, gtol,
                                   prior_normalization=prior_normalization,
                                   itype=itype)

        # Statistics
        (result, istats), (A, R, description), (D, P, I) = iresult.totuple()
        self.__log_inversion__(iroot, iresult.idescription,
                               iresult.residual_vector)


        self.current.istats     = istats
        self.current            = self.current.add_new_child()
        self.current.residuals  = R

        # Update model
        iresult.update_tables(self.current)
        iresult.update_velocity(self.current)
        iresult.update_frechet(self.current)

        # Report
        if report:
            for gid in self.current.vgrids:
                self.__push_report__(gid)

    def report_snapshot(self):
        """
        This function create a snapshot of the current inversion process
        and push a report target on the current stack.

        DEPRECATED

        """
        raise RuntimeError("Deprecated - Use parameter Report = True in push_inversion instead")

    def get_report(self, cssfile):
        """
        This method create the concatenate report from every created snapshots.
        :param cssfile: css style applied to the report.

        :returns: target of the concatenate report
        """
        return self.env.ConcatenateReport([], [cssfile, self.report_targets])

    def get_ray_targets(self, gid):
        """
        Since Rays are not explicitely needed for the inversion process, the
        raytracing is not built normally.

        :param gid:  Grid Identifier

        :returns: list of ray targets
        """
        grid, eik_tgt, fre_tgt, tt_tgt, ray_tgt, gdesc = self.current.vgrids[gid]
        return ray_tgt

    def tomography_qa(self, gid, sigma, ivalue, variation, gtol = 1e-10, root = 'build', itype = 'linearcg'):
        """
        """

        env = self.env
        vgrid = self.current.vgrids[gid]
        grid = vgrid.tgt

        gridfile = os.path.join(root, "qa_initial_%s.pickle" % gid)
        grid = env.HomogenousGridLike([gridfile], [grid,
                                                   env.Value(ivalue)])

        cboard = env.CheckerboardLike([os.path.join(root, 'checkerboard_%s.pickle' % gid)],
                                      [grid, env.Value(sigma), env.Value(ivalue),
                                       env.Value(variation)])

        iroot = os.path.join(root, "qa_%s" % (gid,))
        qbuilder = QATomoQuadraticBuilder(env, iroot, grid, cboard, vgrid, gid, sigma / 2.0)
        qbuilder.process_cross_velocity_inversion()
        iresult = qbuilder.inverse(UniformWeighting(), gtol,
                                   prior_normalization = False, itype = itype)
        node = self.current.add_new_child()

        iresult.update_velocity(node)
        grid = node.vgrids[gid][0]


        return grid

    resolution_test = tomography_qa


InversionPlan2 = InversionPlan # Backward Compatibility


class DBFetchPlan(object):
    def __init__(self, env, database, catalog, evnfile, stafile):
        self.filters = {}
        self.catalog = catalog
        self.db = database
        self.env = env
        self.__evnfile__ = evnfile
        self.__stafile__ = stafile
        self.__evntgt__  = None
        self.__statgt__  = None
        self.builded = False
        if 'Lotic' not in env['TOOLS']:
            env.Tool('Lotic')

    def __check_builded__(self):
        if self.builded:
            raise TypeError("The object is already builded. cannot apply filter anymore")

    def set_cuboid_filter(self, origin, length):
        """
        This method will keeps only the events inside the given cube
        parameters.

        :param origin: 3-tuple representing the origin of the cube.
        :param length: 3-tuple representing the length of each side of the cube

        :returns: None

        """
        self.__check_builded__()
        self.filters['cuboid'] = dbfilter.CuboidFilter(origin, length)

    def set_station_filter(self, sids):
        self.__check_builded__()
        self.filters['station'] = dbfilter.StationFilter(sids)

    def set_event_filter(self, eids):
        self.__check_builded__()
        self.filters['event'] = dbfilter.EventFilter(eids)

    def set_grid_filter(self, grid):
        """
        This method create a cuboid filter from the parameters of the
        grid argument

        :param grid: a GridDescription object

        :returns: None
        """
        self.__check_builded__()
        self.set_cuboid_filter([o + grid.padding for o in grid.origin],
                               [(s - 1) * (grid.spacing) - grid.padding  for s in grid.shape])

    def set_date_filter(self, fromdate, todate):
        """
        This method set the date filter to the given date object.

        :param fromdate: the starting date of the filter (string or datetime)
        :param todate: the starting date of the filter (string or datetime)

        :returns: None

        """
        self.__check_builded__()
        self.filters['date'] = dbfilter.DateFilter(fromdate, todate)

    def additive_gaussian_noise(self, event_time = None, station_time = None,
                                event_position = None, event_file = 'build/noisy_event.pickle',
                                station_file = 'build/noisy_station.pickle'):
        if station_time is not None:
            stafile = self.stafile
            self.__statgt__ = self.env.AdditiveGaussianNoise([station_file],
                                                              [stafile,
                                                               self.env.Value(['delta_t']),
                                                               self.env.Value([station_time])])[0]
        evcol = []
        evstd = []
        if event_time is not None:
            evcol += ['delta_t']
            evstd += [event_time]
        if event_position is not None:
            evcol += ['position']
            evstd += [event_position]
        if len(evcol) != 0:
            evnfile = self.evnfile
            self.__evntgt__ = self.env.AdditiveGaussianNoise([event_file],
                                                              [evnfile,
                                                               self.env.Value(evcol),
                                                               self.env.Value(evstd)])[0]


class H5FDBFetchPlan(DBFetchPlan):
    def __init__(self, env, database, catalog, evnfile, stafile, group):
        DBFetchPlan.__init__(self, env, database, catalog, evnfile, stafile)
        self.group = group

    def __fetch_event__(self):
        if self.__evntgt__ is None:
            self.__evntgt__ = self.env.H5FFetchEvent([self.__evnfile__],
                                                     [self.db, self.env.Value(self.group), self.env.Value(self.catalog),
                                                      self.env.Value(self.filters.values())])[0]
            self.builded = True
        return self.__evntgt__
    evnfile = property(__fetch_event__)
    evntgt = evnfile

    def __fetch_station__(self):
        if self.__statgt__ is None:
            self.__statgt__ = self.env.H5FFetchStation([self.__stafile__],
                                                       [self.db, self.env.Value(self.group), self.env.Value(self.catalog),
                                                        self.env.Value(self.filters.values())])[0]
            self.builded = True
        return self.__statgt__
    stafile = property(__fetch_station__)
    statgt = stafile

    @memoize
    def fetch_traveltime(self, ptype, filetemplate):
        """
        This method creates the traveltime target from the underlying database
        and filters.

        :param ptype: The pick type to use
        :param filetemplate: A string.Template type template for the \
                construction of the splitted traveltime files. A good choice \
                would be something like build/S${sid}.pickle.

        :returns: list of traveltimes targets
        """
        tfilter = {'name' : 'type', 'args' : ptype }
        tt_tgt = self.env.H5FFetchTraveltime([], [self.db, self.stafile, self.evnfile,
                                                  self.env.Value(self.group), self.env.Value(self.catalog),
                                                  self.env.Value(self.filters.values()),
                                                  self.env.Value(ptype),
                                                  self.env.Value(filetemplate)])
        return tt_tgt


class SQLDBFetchPlan(DBFetchPlan):
    """
    This object facilitate the interraction and building of SQL database
    related targets, The events and stations file are built on demand only
    and will have the filters applyed to this object when first requested.
    """

    @memoize
    def fetch_traveltime(self, ptype, filetemplate):
        """
        This method creates the traveltime target from the underlying database
        and filters.

        :param ptype: The pick type to use
        :param filetemplate: A string.Template type template for the \
                construction of the splitted traveltime files. A good choice \
                would be something like build/S${sid}.pickle.

        :returns: list of traveltimes targets
        """
        if (self.evnfile is None) or (self.stafile is None):
            raise RuntimeError("You Should Fetch Event and Station Description "
                               "Before Building Traveltime")

        tfilter = {'name' : 'type', 'args' : ptype }
        tt_tgt = self.env.SQLFetchTraveltime([], [self.db, self.evnfile, self.stafile,
                                                  self.env.Value(self.catalog),
                                                  self.env.Value(self.filters.values() + [tfilter]),
                                                  self.env.Value(filetemplate)])
        return tt_tgt

    @memoize
    def __fetch_event__(self):
        if isinstance(self.__evnfile__, str):
            self.__evnfile__ = self.env.SQLFetchEvent([self.__evnfile__],
                                                      [self.db, self.env.Value(self.catalog),
                                                       self.env.Value(self.filters.values())])[0]
            self.builded = True
        return self.__evnfile__
    evnfile = property(__fetch_event__)

    @memoize
    def __fetch_station__(self):
        if isinstance(self.__stafile__, str):
            self.__stafile__ = self.env.SQLFetchStation([self.__stafile__],
                                                        [self.db, self.env.Value(self.catalog),
                                                         self.env.Value(self.filters.values())])[0]
            self.builded = True
        return self.__stafile__
    stafile = property(__fetch_station__)


class GridDescription(object):
    """
    :param env: The environment where the targets are built
    :param origin: Tuple defining the origin of the grid
    :param shape: Tuple defining the shape of the grid
    :param padding:
    """
    def __init__(self, env, origin, shape, spacing, padding = None):
        self.padding = spacing * 1.5 if padding is None else padding
        if len(origin) != len(shape):
            raise ValueError("Shape and Origin must have the same length")
        self.origin = origin
        self.shape = shape
        self.spacing = spacing

        self.env = env

    def __eq__(self, other):
        return (self.origin == other.origin) \
                and (self.shape == other.shape) \
                and (self.spacing == other.spacing)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.origin, self.shape, self.spacing))

    @staticmethod
    def fromlimits(env, minima, maxima, spacing, padding = None):
        """
        This function should create an optimum grid from a maxima and a minima
        and the spacing for the grid.

        :returns: A GridDescription

        TODO
        """
        pass

    @deprecated
    def transform_to_grid_coordinates(self, output, table):
        """
        TODO : documentation
        :param output: The filename of the transformed grid coordinate
        :param table: The table we wish to convert to the grid coordinate

        :returns: A target grid
        """
        source = [table, self.env.Value(self.origin),
                  self.env.Value(1.0 / self.spacing)]
        target = [output]
        return self.env.TableTransform(target, source)[0]

    def get_checkerboard(self, output, sigma, central_velocity, variation):
        """
        This method create a checkerboard relating with the size and the spacing
        defined in the grid.


        :param output: This is the output file for the checkerboard
        :param sigma: The std deviation of the gaussian for the checkerboard
        :param central_velocity: The median value of the velocity for the \
                checkerboard
        :param variation: The variation in percentage of the central \
                velocity between the lowest and the highest velocity of the grid.

        :returns: A target checkerboard
        """
        env = self.env
        return env.Checkerboard([output],
                                [env.Value(self.shape), env.Value(self.spacing),
                                 env.Value(sigma), env.Value(central_velocity),
                                 env.Value(variation)])

    def get_homogeneous_grid(self, output, value):
        """
        This method creates an homogeneous grid of the size and sacing of the\
        underlying grid.

        :param output: Filename for the output homogeneous grid
        :param value: Filling value of the homogeneous grid

        :returns: An homogeneous grid
        """
        return self.env.HomogenousGrid2([output], [self.env.Value(self),
                                                  self.env.Value(value)])[0]

    def __descr__(self):
        return "<GridDescription origin=%s shape=%s spacing=%d>" % \
                (str(self.origin), str(self.shape), self.spacing)

    def __str__(self):
        return self.__descr__()

    def gen_homogeneous_grid(self, value):
        grid = eikonal.data.EKImageData(self.shape, spacing = self.spacing, origin = self.origin)
        grid.data.fill(value)
        return grid



__doc__ += \
"""
Example Usage : SConstruct
--------------------------
.. note :: This example can be found in the examples directory under the\
        name slopes/SConstruct. To try it just cd in the directory and call
        make.
.. literalinclude:: ../examples/slopes/SConstruct
   :language: python
   :linenos:

"""
