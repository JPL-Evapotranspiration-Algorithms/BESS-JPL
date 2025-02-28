# Using Debian-based python:3.13.2-bookworm base image
FROM python:3.10.16-bookworm as base
ENV HOME /root

FROM base as environment

# Update Debian and install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends tzdata wget fish
RUN pip install --upgrade pip
RUN pip install tomli

FROM environment as installation

# Creating directory inside container to store PGE code
RUN mkdir /app

# Copying current snapshot of PGE code repository into container
COPY . /app/

# Running pip install using pyproject.toml
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app

# FROM dependencies as build
FROM installation as build
RUN pip install -e .[dev]
RUN rm -rvf build; rm -rvf dist; rm -rvf *.egg-info
