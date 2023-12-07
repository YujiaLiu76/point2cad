#!/usr/bin/env bash
set -e
set -x

docker build \
  --pull \
  --tag point2cad \
  -f Dockerfile \
  .
