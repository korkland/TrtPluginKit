#!/bin/bash

CONTAINER_NAME="trt-plugin-kit-container"

# Check if container is already running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container '${CONTAINER_NAME}' is already running."
    echo "Attaching to existing container..."
    docker exec -it ${CONTAINER_NAME} /bin/bash
# Check if container exists but is stopped
elif [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Container '${CONTAINER_NAME}' exists but is stopped."
    echo "Starting and attaching to container..."
    docker start -ai ${CONTAINER_NAME}
else
    echo "Creating and starting new container..."
    # Run the Docker container with GPU support
    docker run \
        --gpus all \
        -it \
        --rm \
        --network host \
        -v $(pwd):/workspace/TrtPluginKit \
        -w /workspace/TrtPluginKit \
        --name ${CONTAINER_NAME} \
        trt-plugin-kit \
        /bin/bash
fi