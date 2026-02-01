-- 021_market_candles_live.sql
BEGIN;

CREATE SCHEMA IF NOT EXISTS market;

-- "Горячая" таблица: хранит только текущие (и последние) свечи, которые WS обновляет UPSERT-ом.
-- Это НЕ историческая таблица. Её размер должен быть небольшим.
CREATE TABLE IF NOT EXISTS market.candles_live (
  symbol          text        NOT NULL,
  timeframe       text        NOT NULL,         -- "1m","5m","15m","1h","4h","1d"
  open_time_ms    bigint      NOT NULL,         -- open time in ms (Binance kline.t)
  close_time_ms   bigint      NOT NULL,         -- close time in ms (kline.T)

  open            double precision NOT NULL,
  high            double precision NOT NULL,
  low             double precision NOT NULL,
  close           double precision NOT NULL,
  volume          double precision NOT NULL,

  trades          integer     NULL,             -- kline.n (если есть)
  is_final        boolean     NOT NULL DEFAULT false, -- kline.x

  last_event_time_ms bigint   NOT NULL,         -- event time (E) or server time
  updated_at      timestamptz NOT NULL DEFAULT now(),
  source          text        NOT NULL DEFAULT 'ws',

  PRIMARY KEY (symbol, timeframe, open_time_ms)
);

-- Индексы под типовые запросы UI/бота: "дай последнюю свечу"
CREATE INDEX IF NOT EXISTS candles_live_tf_time_idx
  ON market.candles_live (timeframe, open_time_ms DESC);

CREATE INDEX IF NOT EXISTS candles_live_symbol_tf_time_idx
  ON market.candles_live (symbol, timeframe, open_time_ms DESC);

-- Мягкая защита от роста (после рестартов): можно чистить всё старше N часов.
-- Вариант 1: просто ручной cleanup по cron/systemd/pg_cron (ниже пример запроса).
-- DELETE FROM market.candles_live WHERE open_time_ms < (extract(epoch from now() - interval '2 days')*1000)::bigint;

COMMIT;