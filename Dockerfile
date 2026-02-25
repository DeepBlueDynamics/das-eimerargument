# Stage 1: Build — CUDA devel image has nvcc for kernel compilation
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04 AS builder

# Install Rust
RUN apt-get update && apt-get install -y curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml build.rs ./
COPY src/ src/
COPY kernels/ kernels/

# Build with GPU support — nvcc compiles kernels to PTX, embedded in binary
RUN cargo build --release --features gpu 2>&1

# Stage 2: Runtime — CUDA runtime has libnvrtc for JIT PTX → CUBIN
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install flask gunicorn && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy simulation binaries
COPY --from=builder /app/target/release/weber-anomaly /usr/local/bin/
COPY --from=builder /app/target/release/brake-recoil /usr/local/bin/
COPY --from=builder /app/target/release/debug-weber /usr/local/bin/
COPY --from=builder /app/target/release/field-residual /usr/local/bin/
COPY --from=builder /app/target/release/shake-test /usr/local/bin/

# Copy data and wrapper
COPY data/ data/
COPY serve.py .

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "--workers", "1", "serve:app"]
