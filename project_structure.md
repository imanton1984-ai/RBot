# Project Structure Plan

Based on the unified design document, here's the planned project structure:

## 1. Crates Structure

### crates/common
- Shared types, enums, and constants
- Time utilities
- Error definitions
- Serialization helpers (rkyv/simd-json)
- Core data structures (Candle, IndicatorState, Signal, etc.)

### crates/api_binance
- Binance API client (REST and WebSocket)
- Authentication and signing
- Rate limiting
- WebSocket connection management
- Error handling for API calls

### crates/ingest
- Market data ingestion service
- WebSocket manager for multiple symbols/TFs
- Candle builder and working candle storage
- Backfill and gap-fill logic
- Health checking and reconnection logic

### crates/compute
- Core computation engine
- Indicator calculation modules (EMA, RSI, MACD, etc.)
- GPU planner for job scheduling
- Raw signal generation
- ML inference integration
- Final scoring logic
- Incremental state management

### crates/db_writer
- Database writer service
- Batch insertion logic
- Idempotency handling
- Migration management
- Retention and compression policies

### crates/order_engine
- Order management service
- Risk management and circuit breakers
- OCO (One-Cancels-Other) implementation
- Traced order management
- Binance execution interface

### crates/position_tracker
- Real-time position tracking
- Signal validity checking
- Dynamic SL/TP management
- Position reconciliation
- Exit decision logic

### crates/api_gateway
- API gateway service
- REST endpoints
- WebSocket streaming
- Authentication
- Health and metrics endpoints

## 2. Infrastructure Structure

### infra/
- docker-compose.yml (TimescaleDB, Redpanda, Grafana, Prometheus)
- timescaledb/init.sql (schema definitions)
- redpanda/topics.sh (topic creation scripts)
- grafana/dashboards/
- prometheus/config.yml

## 3. Scripts Structure

### scripts/
- start.sh (with warmup gates)
- stop.sh
- restart.sh
- healthcheck.sh
- wait_*.sh (various wait scripts)
- preflight.sh (system checks)

## 4. CUDA Structure

### cuda/
- kernels/ (CUDA kernel implementations)
- build.rs (build configuration)
- ptx/ (pre-compiled PTX files)

## 5. ML Structure

### ml/
- trainer/ (training scripts)
- notebooks/ (Jupyter notebooks)
- models/ (exported ONNX models)

## 6. Web Structure

### web/
- package.json (frontend dependencies)
- src/ (React application source)
- api/ (API client)
- components/ (UI components)
- charts/ (charting components)
- store/ (state management)

## 7. Configuration Structure

### config/
- pairs.toml (list of trading pairs)
- runtime.toml (runtime parameters)
- risk.toml (risk management settings)
- ml.toml (ML model configuration)

## 8. Documentation Structure

### docs/
- TradingBot_DesignDoc_Unified.md (main design document)
- architecture.md (detailed architecture)
- api.md (API documentation)
- deployment.md (deployment guide)

## 9. Test Structure

Each crate will contain:
- src/lib.rs (core functionality)
- src/main.rs (binary entry point)
- tests/ (integration tests)
- benches/ (benchmark tests)
- examples/ (usage examples)