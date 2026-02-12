use anyhow::Result;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice, DevicePtr};
use crate::get_cuda_device;

pub struct IndicatorKernelRunner;

impl IndicatorKernelRunner {
    pub fn new() -> Self { Self }

    pub fn calculate_rsi_batch(&self, input: &CudaSlice<f64>, n: usize, period: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "rsi_batch_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32))? };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_sma_batch(&self, input: &CudaSlice<f64>, n: usize, period: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "sma_batch_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32))? };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_ema_batch(&self, input: &CudaSlice<f64>, n: usize, period: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "ema_batch_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32))? };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_bollinger_bands_batch(
        &self, 
        input: &CudaSlice<f64>, 
        n: usize, 
        period: usize, 
        std_dev: f64
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut upper_dev = device.alloc_zeros::<f64>(n)?;
        let mut mid_dev = device.alloc_zeros::<f64>(n)?;
        let mut lower_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "bb_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (input, &mut upper_dev, &mut mid_dev, &mut lower_dev, n as i32, period as i32, std_dev))? 
        };

        Ok((upper_dev, mid_dev, lower_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_macd_batch(
        &self, 
        input: &CudaSlice<f64>, 
        n: usize, 
        fast_period: usize, 
        slow_period: usize, 
        signal_period: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut macd_line_dev = device.alloc_zeros::<f64>(n)?;
        let mut signal_line_dev = device.alloc_zeros::<f64>(n)?;
        let mut histogram_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "macd_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (
                input, 
                &mut macd_line_dev, 
                &mut signal_line_dev, 
                &mut histogram_dev, 
                n as i32, 
                fast_period as i32, 
                slow_period as i32, 
                signal_period as i32
            ))? 
        };

        Ok((macd_line_dev, signal_line_dev, histogram_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_obv_batch(
        &self, 
        close_prices: &CudaSlice<f64>, 
        volumes: &CudaSlice<f64>, 
        n: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "obv_batch_kernel").unwrap();

        unsafe { func.launch(cfg, (close_prices, volumes, &mut output_dev, n as i32))? };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_adx_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "adx_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32))? 
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_atr_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "atr_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32))? 
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_cci_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "cci_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32))? 
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_stochastic_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        k_period: usize, 
        d_period: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut k_dev = device.alloc_zeros::<f64>(n)?;
        let mut d_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "stochastic_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut k_dev, &mut d_dev, n as i32, k_period as i32, d_period as i32))? 
        };

        Ok((k_dev, d_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_vwap_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        volume: &CudaSlice<f64>, 
        n: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "vwap_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, volume, &mut output_dev, n as i32))? 
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_williams_r_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "williams_r_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32))? 
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_alligator_batch(
        &self, 
        source: &CudaSlice<f64>, 
        n: usize, 
        jaw_period: usize, 
        teeth_period: usize, 
        lips_period: usize, 
        jaw_offset: usize, 
        teeth_offset: usize, 
        lips_offset: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut jaw_dev = device.alloc_zeros::<f64>(n)?;
        let mut teeth_dev = device.alloc_zeros::<f64>(n)?;
        let mut lips_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "alligator_batch_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (
                source, 
                &mut jaw_dev, 
                &mut teeth_dev, 
                &mut lips_dev, 
                n as i32, 
                jaw_period as i32, 
                teeth_period as i32, 
                lips_period as i32, 
                jaw_offset as i32, 
                teeth_offset as i32, 
                lips_offset as i32
            ))? 
        };

        Ok((jaw_dev, teeth_dev, lips_dev)) // Return CudaSlices, keeping data on GPU
    }

    // Heuristic predictor methods that work with CudaSlices
    pub fn calculate_rsi_divergence_batch(
        &self, 
        prices: &CudaSlice<f64>, 
        rsi_values: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("predictors", "rsi_divergence_predictor_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (prices, rsi_values, &mut output_dev, n as i32, period as i32))? 
        };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_sr_levels_batch(
        &self, 
        high: &CudaSlice<f64>, 
        low: &CudaSlice<f64>, 
        close: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("predictors", "sr_level_predictor_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32))? 
        };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_momentum_reversal_batch(
        &self, 
        prices: &CudaSlice<f64>, 
        rsi_values: &CudaSlice<f64>, 
        n: usize, 
        period: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<f64>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("predictors", "momentum_reversal_predictor_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (prices, rsi_values, &mut output_dev, n as i32, period as i32))? 
        };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    // Consensus kernel methods
    pub fn run_final_consensus(
        &self, 
        ml_1: &CudaSlice<f32>, 
        ml_2: &CudaSlice<f32>, 
        heur_1: &CudaSlice<f32>, 
        heur_2: &CudaSlice<f32>, 
        n: usize
    ) -> Result<CudaSlice<u8>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let mut output_dev = device.alloc_zeros::<u8>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("consensus", "final_consensus_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (ml_1, ml_2, heur_1, heur_2, &mut output_dev, n as i32))? 
        };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    // Raw signals combiner that works with CudaSlices
    pub fn combine_raw_signals_batch(
        &self,
        rsi_values: &CudaSlice<f64>,
        sma_values: &CudaSlice<f64>,
        ema_values: &CudaSlice<f64>,
        atr_values: &CudaSlice<f64>,
        bb_upper: &CudaSlice<f64>,
        bb_lower: &CudaSlice<f64>,
        bb_mid: &CudaSlice<f64>,
        n: usize,
        num_features: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        
        let total_elements = n * num_features;
        let mut output_dev = device.alloc_zeros::<f64>(total_elements)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("consensus", "raw_signals_combiner_kernel").unwrap();

        unsafe { 
            func.launch(cfg, (
                rsi_values, sma_values, ema_values, atr_values,
                bb_upper, bb_lower, bb_mid,
                &mut output_dev, n as i32, num_features as i32
            ))? 
        };
        
        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_raw_signals_batch(
        &self,
        rsi: &CudaSlice<f64>,
        bb_upper: &CudaSlice<f64>,
        bb_mid: &CudaSlice<f64>,
        bb_lower: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        stoch_k: &CudaSlice<f64>,
        stoch_d: &CudaSlice<f64>,
        atr: &CudaSlice<f64>,
        cci: &CudaSlice<f64>,
        macd_line: &CudaSlice<f64>,
        macd_histogram: &CudaSlice<f64>,
        obv: &CudaSlice<f64>,
        williams_r: &CudaSlice<f64>,
        sma: &CudaSlice<f64>,
        n: usize
    ) -> Result<(CudaSlice<f32>, CudaSlice<i8>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA"))?;

        const NUM_SIGNALS: usize = 10;
        let total_size = n * NUM_SIGNALS;

        let mut scores_dev = device.alloc_zeros::<f32>(total_size)?;
        let mut sides_dev = device.alloc_zeros::<i8>(total_size)?;

        // Collect pointers to all buffers
        let h_ptrs = [
            *rsi.device_ptr(), *bb_upper.device_ptr(), *bb_mid.device_ptr(),
            *bb_lower.device_ptr(), *close.device_ptr(), *stoch_k.device_ptr(),
            *stoch_d.device_ptr(), *atr.device_ptr(), *cci.device_ptr(),
            *macd_line.device_ptr(), *macd_histogram.device_ptr(), *obv.device_ptr(),
            *williams_r.device_ptr(), *sma.device_ptr()
        ];

        // Move this pointer array to the GPU temporarily
        let d_ptrs = device.htod_copy(h_ptrs.to_vec())?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("raw_signals", "calculate_raw_signals_kernel").unwrap();

        unsafe {
            func.launch(cfg, (
                &d_ptrs,
                &mut scores_dev, 
                &mut sides_dev,
                n as i32
            ))?
        };

        Ok((scores_dev, sides_dev))
    }
}