pub mod writers;

use crate::writers::{
    pairs_writer::PairsWriter,
    candles_writer::CandlesWriter,
    indicators_writer::IndicatorsWriter,
    raw_signals_writer::RawSignalsWriter,
    signals_writer::SignalsWriter,
    ml_data_writer::MlDataWriter,
    orders_writer::OrdersWriter,
    position_events_writer::PositionEventsWriter,
};

pub enum WriteTarget {
    Pairs,
    Candles,
    Indicators,
    RawSignals,
    Signals,
    MlData,
    Orders,
    PositionEvents,
}

pub struct DataWriterRouter {
    pairs_writer: PairsWriter,
    candles_writer: CandlesWriter,
    indicators_writer: IndicatorsWriter,
    raw_signals_writer: RawSignalsWriter,
    signals_writer: SignalsWriter,
    ml_data_writer: MlDataWriter,
    orders_writer: OrdersWriter,
    position_events_writer: PositionEventsWriter,
}

impl DataWriterRouter {
    pub fn new() -> Self {
        Self {
            pairs_writer: PairsWriter::new(),
            candles_writer: CandlesWriter::new(),
            indicators_writer: IndicatorsWriter::new(),
            raw_signals_writer: RawSignalsWriter::new(),
            signals_writer: SignalsWriter::new(),
            ml_data_writer: MlDataWriter::new(),
            orders_writer: OrdersWriter::new(),
            position_events_writer: PositionEventsWriter::new(),
        }
    }
    
    pub async fn route_write(&self, target: WriteTarget, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        match target {
            WriteTarget::Pairs => self.pairs_writer.write(data).await,
            WriteTarget::Candles => self.candles_writer.write(data).await,
            WriteTarget::Indicators => self.indicators_writer.write(data).await,
            WriteTarget::RawSignals => self.raw_signals_writer.write(data).await,
            WriteTarget::Signals => self.signals_writer.write(data).await,
            WriteTarget::MlData => self.ml_data_writer.write(data).await,
            WriteTarget::Orders => self.orders_writer.write(data).await,
            WriteTarget::PositionEvents => self.position_events_writer.write(data).await,
        }
    }
}
