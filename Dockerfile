FROM nvcr.io/nvidia/tritonserver:23.08-py3

WORKDIR /workspace
COPY . /workspace

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
