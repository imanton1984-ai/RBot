# Trading Bot v1+ (Rust + CUDA + Redpanda + TimescaleDB + ONNX)

A high-frequency cryptocurrency trading bot for Binance Futures with GPU acceleration, real-time processing, and machine learning capabilities.

## Architecture Overview

This trading bot implements a microservices architecture with the following core components:

### Services
1. **market_ingest** - WebSocket and REST data ingestion
2. **compute_core** - Real-time indicator calculation and signal generation (GPU-first)
3. **db_writer** - Batch database writes to TimescaleDB
4. **order_engine** - Order management and execution
5. **realtime_position_tracker** - Position monitoring and risk management
6. **api_gateway** - Web API and WebSocket interface

### Key Features
- Real-time data processing with sub-second latency
- GPU-accelerated indicator calculations during bootstrap
- Machine learning inference using ONNX models
- Comprehensive risk management and circuit breakers
- Full observability with Prometheus metrics and structured logging
- Docker-based deployment with health checks

## Getting Started

### Prerequisites
- Rust (nightly toolchain)
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for GPU acceleration)
- PostgreSQL and Redpanda

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd trading-bot

# Install Rust toolchain
rustup toolchain install nightly
rustup component add rust-src clippy rustfmt

# Build the project
cargo build --release

# Run infrastructure
docker compose up -d

# Start the bot
./scripts/start.sh
```

## Project Structure
```
trading-bot/
├── Cargo.toml              # Workspace definition
├── rust-toolchain.toml     # Rust toolchain specification
├── .gitignore              # Git ignore rules
├── README.md               # This file
│
├── crates/                 # Rust crates
│   ├── common/             # Shared types and utilities
│   ├── api_binance/        # Binance API integration
│   ├── ingest/             # Data ingestion services
│   ├── compute/            # Indicator calculation and signal generation
│   ├── db_writer/          # Database writer
│   ├── order_engine/       # Order management
│   ├── position_tracker/   # Position tracking
│   └── api_gateway/        # API gateway
│
├── infra/                  # Infrastructure configurations
│   ├── docker-compose.yml  # Docker setup
│   ├── timescaledb/        # Database initialization
│   └── redpanda/           # Message broker setup
│
├── scripts/                # Deployment and management scripts
│   ├── start.sh            # Startup script with health checks
│   ├── stop.sh             # Shutdown script
│   ├── restart.sh          # Restart script
│   └── healthcheck.sh      # Health check script
│
├── cuda/                   # CUDA kernels
│   ├── kernels/            # GPU kernel implementations
│   └── ptx/                # Pre-compiled PTX files
│
├── ml/                     # Machine learning components
│   ├── trainer/            # Training scripts
│   └── notebooks/          # Jupyter notebooks
│
└── web/                    # Web UI (React)
```

## Development

### Building
```bash
# Build for development
cargo build

# Build for release
cargo build --release
```

### Testing
```bash
# Run unit tests
cargo test

# Run benchmarks
cargo bench
```

### Formatting
```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details.