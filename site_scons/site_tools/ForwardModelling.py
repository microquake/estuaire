#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
#
# NOTE :  The output files are separated since the pickling for sparse
#         matrices is not a great as expected and the files generated
#         with it are 5X larger than usual
#
__doc__ = """
"""


from SCons.Script import Builder, Execute, Mkdir

np_load = np.load
np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)

def generate(env):
    import os
    import numpy as np

    import string
    def ForwardModelling(target, source):
        """
        Source :        0 - Event file
                        1 - Station file
                        2 - Traveltime Directory
                        4 - Spacing (Value)
                        5 ... n -  TT files

        Target :        0 - Eikonal Directory
                        0 - Frechet Directory
        """
        evfile, stfile, mdir = [str(s) for s in source[:3]]
        edir, fdir = [str(s) for s in target[:2]]
        gridlst = source[3]
        spacing = source[4]
        ttfiles = [str(s) for s in source[5:]]

        stations = np.load(stfile)

        atargets = []
        for tt in ttfiles:
            stdesc = stations[np.load(tt)['station_id']]
            efile = os.path.join(edir, "arrival_" + os.path.basename(str(tt)))
            ffile = os.path.join(fdir, "frechet_" + os.path.basename(str(tt)))

            pos = env.Value(tuple(stdesc['position'].tolist()))

            etarget = env.EikonalSolver([efile], [gridfile, pos, spacing])[0]
            ftarget = env.Sensivity([ffile], [gridfile, efile, tt])[0]
            atargets.append(ftarget)


        return atargets

    def ForwardModelling2(env,  ttfiles, grid, grid_id, etemplate, ftemplate):
        """
        """
        etemplate = string.Template(etemplate)
        ftemplate = string.Template(ftemplate)

        eik_tgt = []
        fre_tgt = []
        grid_id = os.path.basename(str(grid)) if grid_id is None else grid_id
        for ttf in ttfiles:
            basename = os.path.basename(str(ttf))
            etgt = env.EikonalSolver2([etemplate.substitute(basename = basename)],
                                      [grid, ttf])[0]
            ftgt = env.Sensivity([ftemplate.substitute(basename = basename)],
                                         [grid, etgt, ttf, env.Value(grid_id)])[0]
            eik_tgt.append(etgt)
            fre_tgt.append(ftgt)

        return eik_tgt, fre_tgt

    env.ForwardModelling = ForwardModelling
    env.ForwardModelling2 = ForwardModelling2.__get__(env, env.__class__)
    env.Tool('EikonalSolver')
    env.Tool('Frechet')




def exists(env):
    return 'BuildNow' in env['BUILDERS']
