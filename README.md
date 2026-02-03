# Rust Trading Bot

A high-performance cryptocurrency trading bot built with Rust, featuring real-time market data ingestion, advanced technical analysis, and GPU-accelerated signal processing.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Modules Explanation](#modules-explanation)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Deployment](#deployment)

## Architecture Overview

The trading bot follows a microservice architecture with the following key components:

- **Connections**: Centralized connection management for exchanges, databases, and messaging systems
- **Ingestor**: Real-time and historical market data collection from Binance
- **Compute**: Technical indicator calculations and signal processing (CPU/GPU)
- **Database**: TimescaleDB for time-series market data storage
- **Infrastructure**: Docker containers for PostgreSQL/TimescaleDB and Redpanda/Kafka

## Project Structure

```
whitelist/
├── common/                    # Shared utilities and types
│   └── src/                  # Common data structures, enums, helpers
├── connections/              # Connection management module
│   ├── src/                  # Exchange API clients, WebSocket handlers
│   └── README.md             # Connection module documentation
├── compute/                  # Technical analysis and signal processing
│   ├── indicators/           # Individual indicator implementations (RSI, MACD, etc.)
│   ├── ml/                   # Machine learning components
│   ├── planner/              # Strategy planning logic
│   ├── raw_signals/          # Raw signal generation from indicators
│   ├── scorer/               # Signal scoring algorithms
│   └── src/                  # Main compute service logic
├── config/                   # Configuration files
│   ├── binance.toml          # Binance API settings
│   ├── compute.toml          # Compute engine configuration
│   ├── database.toml         # Database connection settings
│   ├── runtime.toml          # Runtime parameters
│   ├── rust_bot.toml         # Bot-specific settings
│   └── universe.toml         # Trading universe filters
├── cuda/                     # GPU acceleration module
│   ├── kernels/              # CUDA kernel implementations
│   ├── ptx/                  # Precompiled CUDA binaries
│   └── src/                  # CUDA wrapper and management
├── database/                 # Database schema and initialization
│   ├── ddl/                  # SQL schema definitions
│   └── src/                  # Database access layer
├── healthcheck/              # Monitoring and health checking
│   ├── grafana/              # Grafana dashboards
│   └── src/                  # Health check service
├── infra/                    # Infrastructure as code
│   ├── healthcheck/          # Health check infrastructure
│   ├── prometheus/           # Prometheus configuration
│   └── docker-compose.yaml   # Docker orchestration
├── ingestor/                 # Market data ingestion module
│   └── src/                  # Data collection and processing logic
├── logs/                     # Log files directory
├── run/                      # Runtime files (PIDs, etc.)
├── scripts/                  # Utility scripts
│   ├── clean_market_tables.sh # Clean market data tables
│   ├── full_stop.sh          # Full system stop
│   ├── restart.sh            # Restart services
│   ├── start.sh              # Main startup script
│   ├── stop.sh               # Stop services
│   └── test_connection.sh    # Connection testing
├── tests/                    # Test files
├── Cargo.toml                # Workspace configuration
├── Dockerfile                # Container build specification
├── main.rs                   # Entry point for connection testing
└── .env                      # Environment variables
```

## Workflow

### 1. System Startup Process

When the bot starts with `scripts/start.sh`, the following sequence occurs:

#### Infrastructure Initialization
1. **Docker Compose Start**: Launches TimescaleDB (PostgreSQL with time-series extensions) and Redpanda (Kafka-compatible streaming platform)
2. **Service Dependencies**: Waits for database and message broker to become available
3. **Health Checks**: Verifies connectivity to all required services

#### Database Initialization
1. **Schema Creation**: Creates all necessary database tables and hypertables (TimescaleDB)
2. **Index Setup**: Builds optimized indexes for time-series queries
3. **Initial Data**: Populates trading universe with eligible pairs based on volume and liquidity filters

#### Pair Selection Process
1. **Universe Filtering**: Applies filters from `config/universe.toml`:
   - Minimum 24h volume threshold (50M USDT)
   - Excludes stablecoins and commodity-like tokens
   - Allows manual overrides via allowlist/denylist
2. **Exchange Information**: Fetches pair metadata from Binance API
3. **Database Storage**: Stores filtered pairs in `market.pairs` table

#### Data Collection Process
1. **Historical Load**: Fetches historical candles via REST API (`/fapi/v1/klines`)
2. **Real-time Streaming**: Establishes WebSocket connections for live market data
3. **Multi-timeframe Support**: Collects data for multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)

#### Technical Analysis Pipeline
1. **Indicator Calculation**: Computes technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
2. **GPU Acceleration**: Offloads intensive calculations to CUDA when available
3. **Signal Generation**: Creates raw trading signals from indicator values
4. **Scoring**: Assigns confidence scores to generated signals

### 2. Data Flow Architecture

#### Candles Collection (REST + WebSocket)
- **REST API**: Used for historical data backfill (up to 1000 candles per request)
- **WebSocket**: Real-time updates for live candle formation
- **Rate Limiting**: Implements Binance's weight-based rate limiting (40 weight/sec default)
- **Concurrency Control**: Limits simultaneous API requests to prevent rate limit hits

#### Database Storage Strategy
- **TimescaleDB Hypertables**: Optimized partitioning by time intervals
  - 1m candles: Daily partitions
  - 5m candles: 2-day partitions  
  - 15m candles: Weekly partitions
  - 1h candles: Monthly partitions
  - 4h candles: 60-day partitions
  - 1d candles: 90-day partitions
- **Binary Copy**: High-performance bulk insertion using PostgreSQL's COPY protocol
- **Memory Management**: Streams data to prevent memory overflow during large backfills

#### Indicator Calculation Process
1. **Window-Based Processing**: Calculates indicators using configurable lookback windows
2. **Batch Processing**: Groups calculations for efficiency
3. **RAM Storage**: Maintains recent indicator values in memory for quick access
4. **Background Persistence**: Periodically flushes RAM data to database

#### Signal Processing Pipeline
1. **Raw Signal Generation**: Converts indicator values to buy/sell signals
2. **Normalization**: Standardizes signals across different indicators
3. **Scoring**: Applies thresholds and weights to determine signal strength
4. **Aggregation**: Combines multiple signals for final decision-making

### 3. GPU vs CPU Processing

#### CUDA Acceleration
- **Enabled by Default**: When CUDA feature is compiled in
- **Batch Processing**: Processes multiple symbols and timeframes simultaneously
- **Memory Management**: Efficient GPU memory allocation and cleanup
- **Performance**: Significantly faster for mathematical computations

#### CPU Fallback
- **Automatic Switching**: Falls back to CPU when GPU unavailable
- **Consistent Results**: Produces identical results regardless of backend
- **Resource Management**: Adapts to available CPU cores

#### Backend Selection
- **Real-time Processing**: Uses CPU by default for low-latency requirements
- **Historical Backfill**: Uses GPU by default for computational efficiency
- **Configurable**: Can be overridden in `config/compute.toml`

## Modules Explanation

### Common Module
Contains shared data structures, enums, and utility functions used across all services:
- Timeframe definitions (M1, M5, M15, H1, H4, D1)
- Symbol representation and validation
- Configuration loading utilities
- Error types and result wrappers

### Connections Module
Centralized connection management for all external services:
- Binance API client with rate limiting
- PostgreSQL/TimescaleDB connection pooling
- Redpanda/Kafka producer/consumer setup
- WebSocket connection management with automatic reconnection

### Database Module
Database schema management and access layer:
- DDL scripts for table creation
- Migration management
- Connection pooling configuration
- Schema validation and health checks

### Ingestor Module
Market data collection and processing:
- REST API integration for historical data
- WebSocket streaming for real-time updates
- Rate limiting and request scheduling
- Data validation and cleaning
- Binary copy for high-performance insertion

### Compute Module
Technical analysis and signal processing:
- Indicator calculation algorithms (RSI, MACD, EMA, etc.)
- Raw signal generation from indicator values
- Scoring and normalization functions
- Job scheduling and batch processing
- GPU/CPU backend management

### CUDA Module
GPU acceleration for intensive calculations:
- CUDA kernel implementations
- Memory management utilities
- Batch processing optimizations
- Error handling and fallback mechanisms

### Healthcheck Module
System monitoring and health reporting:
- Service health endpoints
- Resource utilization monitoring
- Grafana dashboard configurations
- Alerting mechanisms

## Technical Details

### Database Schema

#### Market Pairs (`market.pairs`)
- `symbol_id`: Primary key (BIGSERIAL)
- `symbol`: Trading pair identifier (e.g., "BTCUSDT")
- `base_asset`: Base currency (e.g., "BTC")
- `quote_asset`: Quote currency (e.g., "USDT")
- `is_active`: Trading eligibility flag
- `volume_24h_usdt`: 24-hour trading volume
- `last_price`: Current market price
- `manual_allow/deny`: Override flags for filtering

#### Candles Tables (`market.candles_*`)
- `time_ms`: Millisecond timestamp (BIGINT)
- `time`: Timestamp as TIMESTAMPTZ
- `symbol_id`: Foreign key reference
- `open`, `high`, `low`, `close`: OHLC values
- `volume`: Trading volume
- TimescaleDB hypertables with time-based partitioning

#### Indicators Tables (`market.indicators_*`)
- `time_ms`: Timestamp reference
- `symbol_id`: Foreign key reference
- Individual indicator columns (ema20, rsi, macd, etc.)
- `features_version`: Schema version tracking
- `updated_at`: Last modification timestamp

#### Raw Signals Table (`market.raw_signals`)
- `time_ms`: Signal timestamp
- `symbol_id`: Reference to trading pair
- `tf_minutes`: Timeframe in minutes
- `indicator_id`: Source indicator identifier
- `side`: Trade direction (1=long, -1=short, 0=neutral)
- `score`: Confidence level (0.0-1.0)
- `details`: Additional signal information as JSONB

### Configuration Parameters

#### Binance API Settings
- `rest_base_url`, `ws_base_url`: API endpoint URLs
- `http_timeout_ms`: Request timeout (8000ms default)
- `rate_limit_soft_rps`: Request rate limiting (39 weight/sec)
- `ws_ping_interval_sec`: WebSocket keepalive (5s)

#### Universe Filters
- `min_quote_volume_usdt_24h`: Minimum volume threshold (50M)
- `exclude_base_assets`: Stablecoin blacklist
- `refresh_interval_sec`: Universe refresh frequency (86400s = 24h)
- `max_change_ratio`: Maximum universe change per refresh (10%)

#### Compute Engine
- `gpu_enabled`: Enable GPU acceleration
- `hot_window_candles`: In-memory window size (1200 candles)
- `gpu_batch_symbols`: Symbols per GPU batch (48)
- `min_indicator_score`: Minimum signal quality (0.60)

### Performance Optimizations

#### Database
- TimescaleDB hypertables for time-series optimization
- Composite indexes on (symbol_id, time) for fast lookups
- Binary copy protocol for bulk insertions
- Connection pooling with 10 connections

#### Memory Management
- Sliding window buffers for recent data
- Background persistence to prevent memory buildup
- Batch processing to minimize allocations

#### Concurrency
- Async/await for I/O operations
- Thread pools for CPU-intensive tasks
- Semaphore-based rate limiting
- Parallel processing of multiple symbols

## Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `KAFKA_BROKERS`: Redpanda/Kafka broker addresses
- `BINANCE_API_KEY`: Binance API key
- `BINANCE_API_SECRET`: Binance API secret

### Configuration Files
Located in `config/` directory:
- `binance.toml`: Exchange-specific settings
- `compute.toml`: Technical analysis parameters
- `database.toml`: Database connection details
- `runtime.toml`: Runtime behavior settings
- `universe.toml`: Trading universe filters

## Deployment

### Prerequisites
- Docker and Docker Compose
- Rust 1.70+ with cargo
- CUDA toolkit (optional, for GPU acceleration)
- At least 8GB RAM recommended

### Startup Process
1. Run `./scripts/start.sh` to initialize infrastructure and start services
2. System will automatically:
   - Start TimescaleDB and Redpanda
   - Initialize database schema
   - Refresh trading universe
   - Start data ingestion
   - Launch compute services

### Service Management
- `./scripts/start.sh`: Start all services
- `./scripts/stop.sh`: Stop all services
- `./scripts/restart.sh`: Restart all services
- `./scripts/full_stop.sh`: Complete shutdown with cleanup

## Potential Duplications Identified

1. **Connection Logic**: Multiple modules may implement similar connection handling - the `connections` module centralizes this to avoid duplication.

2. **Configuration Loading**: Each service loads configuration independently, but uses the shared `common` module to maintain consistency.

3. **Database Queries**: Some basic queries might be duplicated across modules, but are centralized in the `database` module where possible.

4. **Error Handling**: Each module implements its own error types, but follows consistent patterns defined in the `common` module.

## Monitoring and Health Checks

The system includes comprehensive monitoring:
- Grafana dashboards for real-time metrics
- Health check endpoints for each service
- Log aggregation and analysis
- Performance metrics collection
- Resource utilization tracking
