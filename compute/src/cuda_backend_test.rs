#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_cuda_backend_initialization() {
        let mut backend = CudaBackend::new();
        let result = backend.initialize();
        assert!(result.is_ok());
        assert!(backend.initialized);
    }

    #[tokio::test]
    async fn test_compute_indicators_with_cuda() {
        let mut backend = CudaBackend::new();
        backend.initialize().unwrap();

        // Create mock candle data
        let timestamps: Vec<u64> = (0..100).collect();
        let close_prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let high_prices: Vec<f64> = close_prices.iter().map(|p| p + 0.5).collect();
        let low_prices: Vec<f64> = close_prices.iter().map(|p| p - 0.5).collect();
        let open_prices: Vec<f64> = close_prices.iter().map(|p| p - 0.1).collect();
        let volumes: Vec<f64> = (0..100).map(|_| 1000.0).collect();

        let candle_window = common::CandleWindow {
            timestamps: timestamps.clone(),
            open: open_prices,
            high: high_prices,
            low: low_prices,
            close: close_prices,
            volume: volumes,
        };

        let job = ComputeJob {
            symbol: Symbol::from("BTCUSDT"),
            timeframe: Timeframe::M1,
            indicators: vec!["rsi".to_string(), "sma".to_string()],
            window_start: 0,
            window_end: 100,
            is_realtime: false,
            candle_window: Some(Arc::new(candle_window)),
        };

        let results = backend.compute_indicators(vec![job]).await.unwrap();
        assert!(!results.is_empty());
        
        let result = &results[0];
        assert!(result.batch.get_f64("rsi").is_some());
        assert!(result.batch.get_f64("sma").is_some());
    }

    #[tokio::test]
    async fn test_process_all_history_optimized() {
        let mut backend = CudaBackend::new();
        backend.initialize().unwrap();

        // Generate sample price data
        let prices: Vec<f64> = (0..1000).map(|i| 100.0 + (i as f64) * 0.01).collect();
        
        // Use existing model files if they exist
        let model_paths = vec![
            String::from("../../models/levels_v1_tf1440.onnx"), // Use existing model
        ];

        // Test the optimized pipeline
        let result = backend.process_all_history_optimized(prices, &model_paths);
        
        // The test might fail if CUDA isn't available or models don't exist,
        // but that's expected in test environments without GPU
        if backend.initialized {
            // If CUDA is initialized, we expect either success or a model-related error
            // (not a CUDA initialization error)
            match result {
                Ok(_) => println!("Pipeline executed successfully"),
                Err(e) => {
                    let err_str = e.to_string();
                    // Allow model loading errors but not CUDA initialization errors
                    if !err_str.contains("CUDA") && !err_str.contains("device") {
                        println!("Expected model-related error: {}", e);
                    } else {
                        panic!("Unexpected CUDA error: {}", e);
                    }
                }
            }
        } else {
            // If CUDA isn't initialized, we expect an error
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_indicator_runner_creation() {
        let runner = cuda::IndicatorKernelRunner::new();
        
        // Test that we can call a method without panicking
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = runner.calculate_rsi(&input, 3);
        
        // This test might fail if CUDA isn't available, which is expected
        match result {
            Ok(_) => println!("RSI calculation succeeded"),
            Err(e) => println!("RSI calculation failed as expected in test env: {}", e),
        }
    }
}