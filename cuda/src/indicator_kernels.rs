use anyhow::Result;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice, DevicePtr};
use cudarc::driver::DeviceRepr;
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

    // Series-based kernels (new)
    pub fn calculate_rsi_series(&self, input: &CudaSlice<f64>, n: usize, period: usize, batch: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "rsi_wilder_series_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32, batch as i32))? };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_sma_series(&self, input: &CudaSlice<f64>, n: usize, period: usize, batch: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "sma_series_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32, batch as i32))? };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_ema_series(&self, input: &CudaSlice<f64>, n: usize, period: usize, batch: usize) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "ema_series_kernel").unwrap();

        unsafe { func.launch(cfg, (input, &mut output_dev, n as i32, period as i32, batch as i32))? };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_bollinger_bands_series(
        &self,
        input: &CudaSlice<f64>,
        n: usize,
        period: usize,
        std_dev: f64,
        batch: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut upper_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut mid_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut lower_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "bb_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (input, &mut upper_dev, &mut mid_dev, &mut lower_dev, n as i32, period as i32, std_dev, batch as i32))?
        };

        Ok((upper_dev, mid_dev, lower_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_macd_series(
        &self,
        input: &CudaSlice<f64>,
        n: usize,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
        batch: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut macd_line_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut signal_line_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut histogram_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "macd_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (
                input,
                &mut macd_line_dev,
                &mut signal_line_dev,
                &mut histogram_dev,
                n as i32,
                fast_period as i32,
                slow_period as i32,
                signal_period as i32,
                batch as i32
            ))?
        };

        Ok((macd_line_dev, signal_line_dev, histogram_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_obv_series(
        &self,
        close_prices: &CudaSlice<f64>,
        volumes: &CudaSlice<f64>,
        n: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "obv_series_kernel").unwrap();

        unsafe { func.launch(cfg, (close_prices, volumes, &mut output_dev, n as i32, batch as i32))? };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_adx_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        n: usize,
        period: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "adx_wilder_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32, batch as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_atr_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        n: usize,
        period: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "atr_wilder_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32, batch as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_cci_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        n: usize,
        period: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "cci_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32, batch as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_stochastic_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        n: usize,
        k_period: usize,
        d_period: usize,
        batch: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut k_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut d_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "stoch_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, &mut k_dev, &mut d_dev, n as i32, k_period as i32, d_period as i32, batch as i32))?
        };

        Ok((k_dev, d_dev)) // Return CudaSlices, keeping data on GPU
    }

    pub fn calculate_vwap_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        volume: &CudaSlice<f64>,
        n: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "vwap_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, volume, &mut output_dev, n as i32, batch as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_williams_r_series(
        &self,
        high: &CudaSlice<f64>,
        low: &CudaSlice<f64>,
        close: &CudaSlice<f64>,
        n: usize,
        period: usize,
        batch: usize
    ) -> Result<CudaSlice<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "williams_r_series_kernel").unwrap();

        unsafe {
            func.launch(cfg, (high, low, close, &mut output_dev, n as i32, period as i32, batch as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn calculate_alligator_series(
        &self,
        source: &CudaSlice<f64>,
        n: usize,
        jaw_period: usize,
        teeth_period: usize,
        lips_period: usize,
        jaw_offset: usize,
        teeth_offset: usize,
        lips_offset: usize,
        batch: usize
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut jaw_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut teeth_dev = device.alloc_zeros::<f64>(n * batch)?;
        let mut lips_dev = device.alloc_zeros::<f64>(n * batch)?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),  // One block per series
            block_dim: (1, 1, 1),            // One thread per block (since series is sequential)
            shared_mem_bytes: 0,
        };
        let func = device.get_func("indicators", "alligator_series_kernel").unwrap();

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
                lips_offset as i32,
                batch as i32
            ))?
        };

        Ok((jaw_dev, teeth_dev, lips_dev)) // Return CudaSlices, keeping data on GPU
    }

    // Features kernels
    pub fn combine_features_v1(
        &self,
        rsi: &CudaSlice<f32>,
        cci: &CudaSlice<f32>,
        stoch_k: &CudaSlice<f32>,
        stoch_d: &CudaSlice<f32>,
        williams: &CudaSlice<f32>,
        macd: &CudaSlice<f32>,
        macd_signal: &CudaSlice<f32>,
        macd_hist: &CudaSlice<f32>,
        adx: &CudaSlice<f32>,
        sma: &CudaSlice<f32>,
        ema20: &CudaSlice<f32>,
        ema50: &CudaSlice<f32>,
        ema200: &CudaSlice<f32>,
        bb_upper: &CudaSlice<f32>,
        bb_mid: &CudaSlice<f32>,
        bb_lower: &CudaSlice<f32>,
        atr: &CudaSlice<f32>,
        obv: &CudaSlice<f32>,
        vwap: &CudaSlice<f32>,
        volume_spike: &CudaSlice<f32>,
        alligator_jaw: &CudaSlice<f32>,
        alligator_teeth: &CudaSlice<f32>,
        alligator_lips: &CudaSlice<f32>,
        trend: &CudaSlice<f32>,
        trend_short: &CudaSlice<f32>,
        poc: &CudaSlice<f32>,
        n: usize,
        batch: usize
    ) -> Result<CudaSlice<f32>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        // Prepare output buffer
        let total_elements = n * batch * 26; // 26 features
        let output_dev = device.alloc_zeros::<f32>(total_elements)?;

        let total_items = n * batch;
        let threads_per_block = 256;
        let blocks = (total_items + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = device.get_func("features", "combine_features_v1_kernel").unwrap();

        let n_i32: i32 = n as i32;
        let batch_i32: i32 = batch as i32;

        // Use raw args approach to avoid tuple size limits
        use std::ffi::c_void;

        // Order of arguments MUST match the kernel signature
        let mut args: Vec<*mut c_void> = vec![
            rsi.device_ptr().as_kernel_param(),
            cci.device_ptr().as_kernel_param(),
            stoch_k.device_ptr().as_kernel_param(),
            stoch_d.device_ptr().as_kernel_param(),
            williams.device_ptr().as_kernel_param(),
            macd.device_ptr().as_kernel_param(),
            macd_signal.device_ptr().as_kernel_param(),
            macd_hist.device_ptr().as_kernel_param(),
            adx.device_ptr().as_kernel_param(),
            sma.device_ptr().as_kernel_param(),
            ema20.device_ptr().as_kernel_param(),
            ema50.device_ptr().as_kernel_param(),
            ema200.device_ptr().as_kernel_param(),
            bb_upper.device_ptr().as_kernel_param(),
            bb_mid.device_ptr().as_kernel_param(),
            bb_lower.device_ptr().as_kernel_param(),
            atr.device_ptr().as_kernel_param(),
            obv.device_ptr().as_kernel_param(),
            vwap.device_ptr().as_kernel_param(),
            volume_spike.device_ptr().as_kernel_param(),
            alligator_jaw.device_ptr().as_kernel_param(),
            alligator_teeth.device_ptr().as_kernel_param(),
            alligator_lips.device_ptr().as_kernel_param(),
            trend.device_ptr().as_kernel_param(),
            trend_short.device_ptr().as_kernel_param(),
            poc.device_ptr().as_kernel_param(),
            output_dev.device_ptr().as_kernel_param(),
            &n_i32 as *const _ as *mut c_void,
            &batch_i32 as *const _ as *mut c_void,
        ];

        unsafe {
            // Launch using raw args to avoid tuple size limits
            func.launch(cfg, &mut args)?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    // Cast kernels
    pub fn cast_f64_to_f32(&self, input: &CudaSlice<f64>, total: usize) -> Result<CudaSlice<f32>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f32>(total)?;

        let threads_per_block = 256;
        let blocks = (total + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = device.get_func("features", "cast_f64_to_f32_kernel").unwrap();

        unsafe {
            func.launch(cfg, (input, &mut output_dev, total as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
    }

    pub fn fill_const_f32(&self, value: f32, total: usize) -> Result<CudaSlice<f32>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;

        let mut output_dev = device.alloc_zeros::<f32>(total)?;

        let threads_per_block = 256;
        let blocks = (total + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = device.get_func("features", "fill_const_f32_kernel").unwrap();

        unsafe {
            func.launch(cfg, (&mut output_dev, value, total as i32))?
        };

        Ok(output_dev) // Return CudaSlice, keeping data on GPU
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