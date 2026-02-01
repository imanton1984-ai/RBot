
pub struct IndicatorKernelRunner {}

impl IndicatorKernelRunner {
    pub fn new() -> Self {
        Self {}
    }

    pub fn calculate_sma(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let n = input.len();
        let mut output = vec![f64::NAN; n];

        // Calculate SMA using CPU implementation
        for i in 0..n {
            if i < period - 1 {
                continue; // Leave as NaN
            } else {
                let sum: f64 = input[(i + 1 - period)..=i].iter().sum();
                output[i] = sum / period as f64;
            }
        }

        Ok(output)
    }

    pub fn calculate_ema(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let n = input.len();
        let mut output = vec![f64::NAN; n];

        // Calculate EMA using CPU implementation
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Calculate SMA for the first EMA value
        if n >= period {
            let mut sma_sum = 0.0;
            for i in 0..period {
                sma_sum += input[i];
            }
            output[period - 1] = sma_sum / period as f64;

            // Calculate subsequent EMA values
            for i in period..n {
                output[i] = (input[i] - output[i - 1]) * multiplier + output[i - 1];
            }
        }

        Ok(output)
    }

    pub fn calculate_rsi(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let n = input.len();
        let mut output = vec![f64::NAN; n];

        if n < period + 1 {
            return Ok(output);
        }

        // Calculate initial average gain and loss
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in 1..=period {
            let change = input[i] - input[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += change.abs();
            }
        }

        avg_gain /= period as f64;
        avg_loss /= period as f64;

        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            output[period] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            output[period] = 100.0;
        }

        // Calculate subsequent values
        for i in (period + 1)..n {
            let change = input[i] - input[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { change.abs() } else { 0.0 };

            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

            if avg_loss != 0.0 {
                let rs = avg_gain / avg_loss;
                output[i] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                output[i] = 100.0;
            }
        }

        Ok(output)
    }

    pub fn calculate_macd(
        &self,
        input: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        let ema_fast = self.calculate_ema(input, fast_period)?;
        let ema_slow = self.calculate_ema(input, slow_period)?;

        let n = input.len();
        let mut macd_line = vec![f64::NAN; n];
        let mut signal_line = vec![f64::NAN; n];
        let mut histogram = vec![f64::NAN; n];

        for i in 0..n {
            if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
                macd_line[i] = ema_fast[i] - ema_slow[i];
            }
        }

        let signal_ema = self.calculate_ema(&macd_line, signal_period)?;
        signal_line = signal_ema;

        for i in 0..n {
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                histogram[i] = macd_line[i] - signal_line[i];
            }
        }

        Ok((macd_line, signal_line, histogram))
    }

    pub fn calculate_bollinger_bands(
        &self,
        input: &[f64],
        period: usize,
        num_std_dev: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        let n = input.len();
        let mut upper_band = vec![f64::NAN; n];
        let mut middle_band = vec![f64::NAN; n];
        let mut lower_band = vec![f64::NAN; n];

        for i in 0..n {
            if i < period - 1 {
                continue;
            }

            // Calculate SMA (middle band)
            let sum: f64 = input[(i + 1 - period)..=i].iter().sum();
            middle_band[i] = sum / period as f64;

            // Calculate standard deviation
            let mut sum_sq_diff = 0.0;
            for j in (i + 1 - period)..=i {
                let diff = input[j] - middle_band[i];
                sum_sq_diff += diff * diff;
            }
            let std_dev = (sum_sq_diff / period as f64).sqrt();

            // Upper and lower bands
            upper_band[i] = middle_band[i] + (num_std_dev * std_dev);
            lower_band[i] = middle_band[i] - (num_std_dev * std_dev);
        }

        Ok((upper_band, middle_band, lower_band))
    }

    pub fn calculate_obv(
        &self,
        close_prices: &[f64],
        volumes: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let n = close_prices.len();
        if n != volumes.len() {
            return Err("Close prices and volumes arrays must have the same length".into());
        }

        let mut obv = vec![0.0f64; n];
        if n == 0 {
            return Ok(obv);
        }

        obv[0] = volumes[0];

        for i in 1..n {
            if close_prices[i] > close_prices[i - 1] {
                obv[i] = obv[i - 1] + volumes[i];
            } else if close_prices[i] < close_prices[i - 1] {
                obv[i] = obv[i - 1] - volumes[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        Ok(obv)
    }
}