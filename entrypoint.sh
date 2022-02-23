#!/bin/bash
source /venv/estuaire/bin/activate
exec "$@"
export PYTHONPATH=$PYTHONPATH:/estuaire/site_scons