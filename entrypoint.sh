#!/bin/bash
source /venv/estuaire/bin/activate
exec "$@"
export PYTHONPATH=$PYTHONPATH:/agstd:/eikonal:/estuaire/site_scons