#!/bin/bash

HERE=$(dirname "$(readlink -f $0)")

docker build -t triton-vad-server:latest -f "${HERE}"/Dockerfile "${HERE}"