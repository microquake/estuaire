FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y python3.9 python3-pip python3.9-dev \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && apt install -y curl \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && apt install python3.9-venv \
    && python -m venv /venv/estuaire \
    && source /venv/estuaire/bin/activate \
    && pip install poetry \
    && poetry config virtualenvs.create false --local \
    && apt install git \
    && git clone https://github.com/microquake/estuaire.git \
    & apt remove git \
    && apt remove curl \
    && python get-pip.py \


  && apt-get install -y python3.9-pip python3.9-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip install poetry