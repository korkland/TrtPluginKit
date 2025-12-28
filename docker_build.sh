#!/bin/bash

# Build the Docker image with proxy settings
docker build \
    -t trt-plugin-kit \
    .

echo "Docker image 'trt-plugin-kit' built successfully!"