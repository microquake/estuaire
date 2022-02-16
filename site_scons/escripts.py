import logger
import subprocess

DOUBLE_DIFFERENCE   = "double_difference"
CONJUGATE_GRADIENT  = "conjugate_gradient"
EIKONAL_SOLVER      = "eikonal_solver"
RAYTRACE            = "raytrace"
FRECHET_DERIVATIVES = "sensitivity"
CHECKERBOARD        = "checkerboard"
CROSS_SENSIVITY     = "cross_sensivity"
EXPORT_TXT_GRID     = "export_txt_grid"

scriptlst = [DOUBLE_DIFFERENCE, CONJUGATE_GRADIENT, EIKONAL_SOLVER, RAYTRACE,
             FRECHET_DERIVATIVES, CHECKERBOARD, CROSS_SENSIVITY,
             EXPORT_TXT_GRID]

for script in scriptlst:
    if logger.tools.isEnabledFor(logger.DEBUG):
        logger.tools.debug("Verifying the presence of : %s script" % script)
    if subprocess.Popen(['which', script], stdout=subprocess.PIPE).wait() != 0:
        logger.tools.critical("%s script from eikonal-ng could not be found"
                              "in the current path" % script)
        raise EnvironmentError("Path incorrect. Please advise...")
