-- 001_extensions_and_schemas.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS trade;

-- (опционально) отдельная схема для сервисных вещей
CREATE SCHEMA IF NOT EXISTS core;