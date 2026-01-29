# Connections Module

This module handles all external connections for the trading bot, including:

- Binance API connection
- Binance WebSocket connection
- Redpanda (Kafka) connection
- Database connection

## Features

- **Stable Connections**: Optimized for minimal latency and maximum stability
- **Health Checks**: Built-in connection monitoring and health checks
- **Environment Configuration**: Configurable via environment variables
- **Async Support**: Full async/await support for high-performance operations

## Components

### Binance API (`binance_api.rs`)
Handles REST API communication with Binance exchange:
- Exchange information retrieval
- K-line/candlestick data fetching
- Ping/keep-alive functionality

### Binance WebSocket (`binance_websocket.rs`)
Manages WebSocket connections for real-time market data:
- Multiple stream subscriptions
- Ping/pong handling for connection stability
- Message parsing and routing

### Redpanda (`redpanda.rs`)
Handles messaging via Redpanda (compatible with Kafka):
- Producer and consumer setup
- Topic management
- Message publishing and consumption

### Database (`database.rs`)
Manages PostgreSQL/TimescaleDB connections:
- Connection pooling
- Query execution
- Health monitoring

### Connection Checker (`connection_check.rs`)
Provides health checks for all connections:
- Comprehensive status reporting
- Response time measurements
- Multi-service monitoring

## Environment Variables

Configuration is handled via the `.env` file with the following variables:

```env
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=timescaledb_binance 
DB_PORT=5433 
DATABASE_URL=postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance 

# Redpanda Infrastructure
RP_KAFKA_PORT=19092 
RP_ADMIN_PORT=9644 
RP_CONSOLE_PORT=8080 

# App Settings
KAFKA_BROKERS=127.0.0.1:19092 

# Service Ports
MARKET_INGEST_PORT=9001 
SVC_HEALTH_PORT=9005 
COMPUTE_CORE_PORT=9010 
API_GATEWAY_PORT=9080 
ORDER_ENGINE_PORT=9020 
POSITION_TRACKER_PORT=9030 
WEBGUI_PORT=9040

# Kafka/Redpanda Topics
TOPIC_CANDLES_CLOSE=candles.close
TOPIC_INDICATORS_CLOSE=indicators.close
TOPIC_CANDLES_LIVE=candles.live
TOPIC_PAIRS=market.pairs
TOPIC_RAW_SIGNALS=trade.raw_signals

# Consumer Group
WRITER_GROUP=writer
```

## Usage

To run connection tests:

```bash
cargo run
```

To run specific tests:

```bash
cargo test