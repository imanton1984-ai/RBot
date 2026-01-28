-- Test data tables for integration testing
CREATE TABLE IF NOT EXISTS integration_test_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timestamp BIGINT NOT NULL,
    trade_id BIGINT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cross_system_test (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timestamp BIGINT NOT NULL,
    trade_id BIGINT UNIQUE NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_integration_test_timestamp ON integration_test_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_cross_system_test_timestamp ON cross_system_test(timestamp);
CREATE INDEX IF NOT EXISTS idx_integration_symbol ON integration_test_data(symbol);
CREATE INDEX IF NOT EXISTS idx_cross_system_symbol ON cross_system_test(symbol);