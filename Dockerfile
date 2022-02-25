FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

#RUN apt update \
#    && apt install python2 -y \

COPY . /estuaire

RUN apt-get update \
    && apt-get install -y python3.9 python3-pip python3.9-dev \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && apt install python3.9-venv \
    && python3.9 -m venv /venv/estuaire \
    && . /venv/estuaire/bin/activate \
    && pip install poetry \
    && poetry config virtualenvs.create false --local \
    && apt install git -y

RUN . /venv/estuaire/bin/activate \
    && cd /estuaire && git submodule init && poetry install\
    && cd libraries/eikonal && make clean && make \
    && python setup.py install \
    && cp bin/* /venv/estuaire/bin/. \
    && cd ../.. && poetry install \
    && cd /estuaire \
    && chmod +x entrypoint.sh

ENV PYTHONPATH=$PYTHONPATH:/estuaire/site_scons

ENTRYPOINT ["./estuaire/entrypoint.sh"]