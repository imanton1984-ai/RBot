# Trading Bot Connection System

This project provides a robust, high-performance connection system for a trading bot that interfaces with Binance exchange, processes market data, and stores it in a database.

## Architecture Overview

The system consists of multiple interconnected components:

- **Binance API Client**: Fetches historical market data
- **Binance WebSocket Client**: Receives live market data streams
- **Redpanda (Kafka)**: Message streaming for real-time data processing
- **PostgreSQL/TimescaleDB**: Storage for market data (pairs, candles, indicators)
- **Connection Management**: Health checks and monitoring

## System Components

### 1. Data Flow Architecture
```
Binance API → Historical Data → Kafka → Database
Binance WebSocket → Live Data → Kafka → Database
```

### 2. Connection Modules
- `binance_api.rs`: Handles REST API communication with rate limiting
- `binance_websocket.rs`: Manages WebSocket connections for real-time data
- `redpanda.rs`: Kafka-based message streaming for data processing
- `database.rs`: PostgreSQL/TimescaleDB connection with connection pooling
- `connection_check.rs`: Health monitoring and status reporting

## Scripts

### Control Scripts (in `scripts/` directory)

1. **`start.sh`** - Starts the entire system:
   - Loads environment variables from `.env`
   - Builds Rust binaries
   - Starts Docker containers (if available)
   - Initializes database schema
   - Starts the main application service

2. **`stop.sh`** - Stops all services:
   - Terminates application processes
   - Stops Docker containers
   - Performs cleanup (removes unused Docker resources)
   - Runs `cargo clean`

3. **`restart.sh`** - Performs stop and start sequence

4. **`test_connection.sh`** - Runs comprehensive connection tests:
   - Tests all connection modules
   - Reports performance metrics
   - Logs connection status and response times

## Project Structure

```
├── .env                    # Environment variables
├── main.rs                 # Main application entry point
├── Cargo.toml              # Rust project configuration
├── connections/            # Connection modules library
│   ├── Cargo.toml          # Dependencies for connection modules
│   └── src/                # Source code for connection modules
├── database/               # Database initialization scripts
│   └── ddl/                # Database schema definitions
├── scripts/                # Control scripts
│   ├── start.sh           # Start all services
│   ├── stop.sh            # Stop all services  
│   ├── restart.sh         # Restart services
│   └── test_connection.sh # Run connection tests
├── tests/                  # Integration and unit tests
└── logs/                   # Runtime logs
```

## Environment Configuration

The `.env` file contains all necessary configuration:

- Database connection settings
- Kafka/Redpanda broker addresses
- Service ports and timeouts
- Binance API credentials (if needed)

## Build and Run

### Prerequisites
- Rust toolchain
- Docker (optional, for containerized services)
- PostgreSQL/TimescaleDB server

### Running the System

1. **Setup environment**: Ensure `.env` file is properly configured
2. **Start services**: `./scripts/start.sh`
3. **Run tests**: `./scripts/test_connection.sh`
4. **Stop services**: `./scripts/stop.sh`

## Data Pipeline

### Historical Data Flow
1. Binance API fetches historical market data
2. Data is sent through Kafka/Redpanda
3. Data is stored in TimescaleDB tables

### Live Data Flow
1. Binance WebSocket receives real-time market data
2. Data is sent through Kafka/Redpanda
3. Data is stored in TimescaleDB tables

## Testing

The system includes comprehensive testing:
- Unit tests for each connection module
- Integration tests for cross-module functionality
- Stress tests for performance evaluation
- Connection tests for health monitoring

## Security and Reliability

- Connection pooling for database efficiency
- Rate limiting for exchange API compliance
- WebSocket ping/pong for connection stability
- Health checks and monitoring
- Graceful error handling and recovery