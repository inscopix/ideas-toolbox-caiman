# Builder stage to create venv with analysis dependencies
FROM public.ecr.aws/lts/ubuntu:22.04 AS builder

# Create ideas home dir to save venv for analysis
WORKDIR /ideas

RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential g++ \
    && update-ca-certificates

# Create a venv with uv to install python dependencies
# This can be done globally, but using venv is best practice

# ========================== Python Dependency Installation with uv ===========================

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Disable bytecode compilation (causes too many files open OS error)
ENV UV_COMPILE_BYTECODE=0
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Install venv into in the ideas user's home
ENV UV_PROJECT_ENVIRONMENT=/ideas/venv VIRTUAL_ENV=/ideas/venv
# Allow uv to download required python version based on pinned .python-version
ENV UV_PYTHON_DOWNLOADS=automatic
# Strictly use frozen dependencies, and require cryptographic verification of each package
ENV UV_LOCKED=1 UV_REQUIRE_HASHES=1 UV_VERIFY_HASHES=1
ENV UV_CACHE_DIR=/tmp/.cache/uv
# Specify accessible directory to install python
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
    --mount=type=cache,id=uv-cache-ideas,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=resources/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl,target=resources/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl \
    uv sync --no-install-project --group analysis \
    # fix for pynwb api which requires write permissions to site-package dir
    && /ideas/venv/bin/python -c "import pynwb" 

# Base image for production use
FROM public.ecr.aws/lts/ubuntu:22.04 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Arguments for python installation
ARG PYTHON=python3.11
ARG VENV=venv
ARG PYTHON_VENV=/ideas/${VENV}/bin/python

# Create ideas user
# This is no longer necessary to do, but good practice anyways
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

# Create ideas home dir
WORKDIR /ideas

# ========================== Apt Dependency Installation for Runtime ===========================

RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        # Install and update ca certs for security
        ca-certificates \
        libglib2.0-0 \
        libgl1-mesa-glx \
        ffmpeg \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ========================== Python Dependencies Copied from Build Stage ===========================

COPY --from=builder --chown=ideas /ideas/venv /ideas/venv
COPY --from=builder --chown=ideas /opt/uv-python /opt/uv-python

USER ideas

# Add venv bin to path
ENV PATH="/ideas/venv/bin:${PATH}"

CMD ["/bin/bash"]

# Create image for testing which copies tool code and test data to
# docker image in order to facilitate unit testing in an isolated environment.
# This can also be acheived with volume mounts, but that can clutter up
# your local folder with files generated during testing.
FROM base AS test

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=0
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Install venv into in the ideas user's home
ENV UV_PROJECT_ENVIRONMENT=/ideas/venv VIRTUAL_ENV=/ideas/venv
# Allow uv to provision the pinned Python version required by the lockfile.
ENV UV_PYTHON_DOWNLOADS=automatic
# Strictly use frozen dependencies, and require cryptographic verification of each package
ENV UV_LOCKED=1 UV_REQUIRE_HASHES=1 UV_VERIFY_HASHES=1
ENV UV_CACHE_DIR=/tmp/.cache/uv

USER root

RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
    --mount=type=cache,id=uv-cache-ideas,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=resources/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl,target=resources/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl \
    uv sync --no-install-project --group analysis --group test

COPY --chown=ideas ./ /ideas
COPY "resources/cnn_model*" ./

USER ideas

# Vulnerability scanning stage using Trivy
# Copies the runtime filesystem to a subdirectory to avoid overwrite conflicts with the trivy base image
FROM base AS scanner

USER root

# Install ca certs so trivy can download appropriate repositories
RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=from=aquasec/trivy:0.69.3,source=/usr/local/bin/trivy,target=/usr/local/bin/trivy \
    trivy rootfs --no-progress --ignore-unfixed --skip-files /usr/local/bin/trivy --severity CRITICAL,HIGH --exit-code 1 / \
    && touch /scan-ok

# Final stage - identical to runtime but depends on successful scan
FROM base AS final

# COPY --from=scanner /scan-ok /scan-ok
