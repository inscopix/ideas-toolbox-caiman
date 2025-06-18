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

ENV PATH="${PATH}:/ideas/.local/bin"

ARG PACKAGE_REQS
ARG PYTHON_VERSION=3.10.0
ARG PYTHON=python3.10
ENV PACKAGE_REQS=$PACKAGE_REQS

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
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ${PYTHON} -m pip install --no-cache-dir awscli==1.35.17 boto3==1.35.17 click requests Cython

# Link python to specific version
RUN ln -s /usr/bin/${PYTHON} /usr/bin/python

# Copy files needed by the toolbox
COPY setup.py function_caller.py user_deps.txt pytest.ini install_imported_code.sh "resources/*" ./

# Install Python packages
RUN ${PYTHON} -m pip install --default-timeout=1000 -e . \
    && ${PYTHON} -m pip install *.whl

# Install user code from git repo if needed
RUN /bin/bash install_imported_code.sh

# Change owner of directories to the ideas user
COPY --chown=ideas toolbox /ideas/toolbox
COPY --chown=ideas commands /ideas/commands

# Mark commands as executable (return 0 even if there are no files in /ideas/commands)
RUN chmod +x /ideas/commands/* ; return 0

# Copy toolbox info and annotation files
COPY --chown=ideas info /ideas/info

USER ideas
CMD ["/bin/bash"]
