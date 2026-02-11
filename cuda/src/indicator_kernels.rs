use anyhow::Result;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use crate::get_cuda_device;

pub struct IndicatorKernelRunner;

impl IndicatorKernelRunner {
    pub fn new() -> Self { Self }

    pub fn calculate_rsi(&self, input: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = input.len();
        if n == 0 { return Ok(vec![]); }

        let inp_dev = device.htod_copy(input.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "rsi_kernel").unwrap();
        
        unsafe { func.launch(cfg, (&inp_dev, &mut out_dev, n as i32, period as i32)) }?;

        let result = device.dtoh_sync_copy(&out_dev)?;
        Ok(result)
    }

    pub fn calculate_sma(&self, input: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = input.len();
        if n == 0 { return Ok(vec![]); }
        
        let inp_dev = device.htod_copy(input.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "sma_kernel").unwrap();
        
        unsafe { func.launch(cfg, (&inp_dev, &mut out_dev, n as i32, period as i32)) }?;
        
        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_ema(&self, input: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = input.len();
        if n == 0 { return Ok(vec![]); }

        let inp_dev = device.htod_copy(input.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "ema_kernel").unwrap();

        unsafe { func.launch(cfg, (&inp_dev, &mut out_dev, n as i32, period as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }
    
    pub fn calculate_bollinger_bands(&self, input: &[f64], period: usize, std_dev: f64) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = input.len();
        if n == 0 { return Ok((vec![], vec![], vec![])); }

        let inp_dev = device.htod_copy(input.to_vec())?;
        let mut upper_dev = device.alloc_zeros::<f64>(n)?;
        let mut mid_dev = device.alloc_zeros::<f64>(n)?;
        let mut lower_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "bb_kernel").unwrap();

        unsafe { func.launch(cfg, (&inp_dev, &mut upper_dev, &mut mid_dev, &mut lower_dev, n as i32, period as i32, std_dev)) }?;

        let upper = device.dtoh_sync_copy(&upper_dev)?;
        let mid = device.dtoh_sync_copy(&mid_dev)?;
        let lower = device.dtoh_sync_copy(&lower_dev)?;

        Ok((upper, mid, lower))
    }

    pub fn calculate_macd(&self, input: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = input.len();
        if n == 0 { return Ok((vec![], vec![], vec![])); }

        let inp_dev = device.htod_copy(input.to_vec())?;
        let mut macd_line_dev = device.alloc_zeros::<f64>(n)?;
        let mut signal_line_dev = device.alloc_zeros::<f64>(n)?;
        let mut histogram_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "macd_kernel").unwrap();

        unsafe { func.launch(cfg, (&inp_dev, &mut macd_line_dev, &mut signal_line_dev, &mut histogram_dev, n as i32, fast_period as i32, slow_period as i32, signal_period as i32)) }?;

        let macd_line = device.dtoh_sync_copy(&macd_line_dev)?;
        let signal_line = device.dtoh_sync_copy(&signal_line_dev)?;
        let histogram = device.dtoh_sync_copy(&histogram_dev)?;

        Ok((macd_line, signal_line, histogram))
    }

    pub fn calculate_obv(&self, close_prices: &[f64], volumes: &[f64]) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = close_prices.len();
        if n == 0 { return Ok(vec![]); }

        let close_dev = device.htod_copy(close_prices.to_vec())?;
        let vol_dev = device.htod_copy(volumes.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "obv_kernel").unwrap();

        unsafe { func.launch(cfg, (&close_dev, &vol_dev, &mut out_dev, n as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_adx(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok(vec![]); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "adx_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &mut out_dev, n as i32, period as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok(vec![]); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "atr_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &mut out_dev, n as i32, period as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_cci(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok(vec![]); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "cci_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &mut out_dev, n as i32, period as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_stochastic(&self, high: &[f64], low: &[f64], close: &[f64], k_period: usize, d_period: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok((vec![], vec![])); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let mut k_dev = device.alloc_zeros::<f64>(n)?;
        let mut d_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "stochastic_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &mut k_dev, &mut d_dev, n as i32, k_period as i32, d_period as i32)) }?;

        let k_values = device.dtoh_sync_copy(&k_dev)?;
        let d_values = device.dtoh_sync_copy(&d_dev)?;
        
        Ok((k_values, d_values))
    }

    pub fn calculate_vwap(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok(vec![]); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let vol_dev = device.htod_copy(volume.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "vwap_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &vol_dev, &mut out_dev, n as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_williams_r(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = high.len();
        if n == 0 { return Ok(vec![]); }

        let high_dev = device.htod_copy(high.to_vec())?;
        let low_dev = device.htod_copy(low.to_vec())?;
        let close_dev = device.htod_copy(close.to_vec())?;
        let mut out_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "williams_r_kernel").unwrap();

        unsafe { func.launch(cfg, (&high_dev, &low_dev, &close_dev, &mut out_dev, n as i32, period as i32)) }?;

        Ok(device.dtoh_sync_copy(&out_dev)?)
    }

    pub fn calculate_alligator(&self, source: &[f64], jaw_period: usize, teeth_period: usize, lips_period: usize, jaw_offset: usize, teeth_offset: usize, lips_offset: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let device = get_cuda_device().ok_or(anyhow::anyhow!("No CUDA device"))?;
        let n = source.len();
        if n == 0 { return Ok((vec![], vec![], vec![])); }

        let source_dev = device.htod_copy(source.to_vec())?;
        let mut jaw_dev = device.alloc_zeros::<f64>(n)?;
        let mut teeth_dev = device.alloc_zeros::<f64>(n)?;
        let mut lips_dev = device.alloc_zeros::<f64>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = device.get_func("indicators", "alligator_kernel").unwrap();

        unsafe { func.launch(cfg, (&source_dev, &mut jaw_dev, &mut teeth_dev, &mut lips_dev, n as i32, jaw_period as i32, teeth_period as i32, lips_period as i32, jaw_offset as i32, teeth_offset as i32, lips_offset as i32)) }?;

        let jaw = device.dtoh_sync_copy(&jaw_dev)?;
        let teeth = device.dtoh_sync_copy(&teeth_dev)?;
        let lips = device.dtoh_sync_copy(&lips_dev)?;

        Ok((jaw, teeth, lips))
    }
}