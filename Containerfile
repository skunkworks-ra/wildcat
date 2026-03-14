FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04

# System dependencies (devel image includes CUDA headers for llama.cpp build)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    cmake \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    libopenblas-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source with CUDA support.
# libcuda.so.1 is not in the devel image (host driver injects it at runtime).
# Symlink the versioned stub so the linker finds it, and pass the stubs dir
# explicitly via cmake linker flags — LIBRARY_PATH alone doesn't reach cmake.
# Clone first — cached independently so tweaking cmake flags doesn't re-clone
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp

# Static build (-DBUILD_SHARED_LIBS=off) avoids the libcuda.so.1 link-time
# problem entirely — the binary is self-contained and picks up the host driver
# at runtime via --device nvidia.com/gpu=all
RUN cmake -B /opt/llama.cpp/build -S /opt/llama.cpp \
        -DGGML_CUDA=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/llama.cpp/build --target llama-server -j$(nproc) \
    && rm -rf /opt/llama.cpp/.git
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# Python virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# wildcat — install first so ms-inspect deps layer on top
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

# ms-inspect is volume-mounted at runtime (/opt/ms-inspect).
# entrypoint.sh installs it before supervisord starts.
RUN mkdir -p /opt/ms-inspect /skills /data/jobs /models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY supervisord.conf /etc/supervisord.conf

EXPOSE 8000 8080 8081

ENTRYPOINT ["/entrypoint.sh"]
