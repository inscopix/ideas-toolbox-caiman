FROM public.ecr.aws/lts/ubuntu:22.04 AS base

# General env variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TQDM_DISABLE=1

# CaImAn env variables
# https://caiman.readthedocs.io/en/latest/Installation.html#section-4c-setting-up-environment-variables
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

ARG PYTHON_VERSION=3.10.0
ARG PYTHON=python3.10

# Create ideas user
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

WORKDIR /ideas

# Install dependencies
RUN apt update && apt upgrade -y \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt install -y gcc python3-dev \
    && apt install -y python3.10 python3.10-dev python3-pip python3.10-distutils git curl libgl1 ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy files needed by the toolbox
COPY pyproject.toml "resources/*" ./

# Install Python packages
RUN ${PYTHON} -m pip install --default-timeout=1000 .[dev] && ${PYTHON} -m pip install *.whl

# Link python to specific version
RUN ln -s /usr/bin/${PYTHON} /usr/bin/python

USER ideas
CMD ["/bin/bash"]
