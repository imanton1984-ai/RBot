-- database/ddl/031_add_new_indicators.sql
-- Migration: Add new indicator columns to market.indicators_wide
-- Indicators: MFI, Fibonacci Pivot/R1/S1, SuperTrend, CMF
--
-- Safe to run multiple times (IF NOT EXISTS pattern via DO blocks)

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'mfi') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN mfi REAL;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'fibo_pivot') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN fibo_pivot REAL;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'fibo_r1') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN fibo_r1 REAL;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'fibo_s1') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN fibo_s1 REAL;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'supertrend') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN supertrend REAL;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'supertrend_dir') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN supertrend_dir SMALLINT;
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'market' AND table_name = 'indicators_wide' AND column_name = 'cmf') THEN
        ALTER TABLE market.indicators_wide ADD COLUMN cmf REAL;
    END IF;
END $$;
