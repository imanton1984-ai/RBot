-- tf_minutes только из набора
ALTER TABLE market.raw_signals
  ADD CONSTRAINT chk_raw_tf
  CHECK (tf_minutes IN (1,5,15,60,240,1440));

ALTER TABLE trade.final_signals
  ADD CONSTRAINT chk_final_tf
  CHECK (tf_minutes IN (1,5,15,60,240,1440));

-- side only -1 or 1 (в финальных сигналов)
ALTER TABLE trade.final_signals
  ADD CONSTRAINT chk_final_side
  CHECK (side IN (-1, 1));

-- score range
ALTER TABLE market.raw_signals
  ADD CONSTRAINT chk_raw_score
  CHECK (score >= 0 AND score <= 1);

ALTER TABLE trade.final_signals
  ADD CONSTRAINT chk_final_score
  CHECK (final_score >= 0 AND final_score <= 1);

-- indicator_id range (expanded to accommodate more indicators)
ALTER TABLE market.raw_signals
  ADD CONSTRAINT chk_indicator_id
  CHECK (indicator_id BETWEEN 1 AND 20);

-- signal_kind range (expanded to accommodate more signal types)
ALTER TABLE market.raw_signals
  ADD CONSTRAINT chk_signal_kind
  CHECK (signal_kind BETWEEN 1 AND 5);
