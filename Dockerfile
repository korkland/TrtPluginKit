FROM nvcr.io/nvidia/tensorrt:25.10-py3

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

RUN apt-get update && apt-get install -y \
    cmake \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX first
RUN pip3 install --no-cache-dir onnx onnxscript

# Install torch-tensorrt FIRST - it will pull the correct torch version
RUN pip3 install --no-cache-dir torch-tensorrt==2.9.0

# Now install torchvision that matches the torch version installed by torch-tensorrt
# torch-tensorrt 2.9.0 will install torch 2.9.x, so we install matching torchvision
RUN pip3 install --no-cache-dir torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Optional: Install modelopt if you need quantization support
RUN pip3 install --no-cache-dir "nvidia-modelopt[torch]>=0.11.0"

# Set working directory
WORKDIR /workspace/TrtPluginKit

# Copy and build project
COPY . .
RUN mkdir -p build && cd build && \
    cmake .. && \
    cmake --build . -j$(nproc)

CMD ["/bin/bash"]