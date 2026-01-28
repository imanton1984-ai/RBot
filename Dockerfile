FROM rust:1.93-slim-bookworm AS builder
WORKDIR /app

# deps for rdkafka (librdkafka) + tls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    pkg-config \
    cmake \
    make \
    g++ \
    git \
    libssl-dev \
    libsasl2-dev \
  && rm -rf /var/lib/apt/lists/*

# manifests first (for caching)
COPY Cargo.toml Cargo.lock ./
COPY common/Cargo.toml common/Cargo.toml
COPY connections/Cargo.toml connections/Cargo.toml
COPY database/Cargo.toml database/Cargo.toml
COPY healthcheck/Cargo.toml healthcheck/Cargo.toml

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch

# sources
COPY common common
COPY connections connections
COPY database database
COPY healthcheck healthcheck

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release -p connections --bin connections

RUN strip /app/target/release/connections || true

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/connections /usr/local/bin/connections

# non-root
USER 65532:65532

ENTRYPOINT ["/usr/local/bin/connections"]
