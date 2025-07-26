# Docker image
# https://hub.docker.com/layers/library/python/3.10.12-slim/images/sha256-0549f89d5387b0df640f136594a1db3a2aaa8cfbf351ec988d07325f8997b77a?context=explore
FROM python:3.12.10-slim

RUN apt -y update && apt -y upgrade
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt -y install --no-install-recommends \
    # Necessary for pip install
    gcc g++ libc-dev \
    # For development and debug (procps: `free -h`)
    bash curl git net-tools vim procps \
    # # For pyvirtualdisplay, gym-notebook-wrapper
    # xvfb \
    # # For mpi4py
    # cmake libopenmpi-dev zlib1g-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


# for updating python module
RUN pip install --upgrade pip

ARG WORKING_DIRECTORY
WORKDIR /${WORKING_DIRECTORY}

# Install pip modules
COPY ./docker/pip_env/requirements.txt ./
RUN pip install -r requirements.txt
