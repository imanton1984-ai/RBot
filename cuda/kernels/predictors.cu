#include <cuda_runtime.h>
#include <stdio.h>

// Heuristic predictor kernel for RSI divergence detection
__global__ void rsi_divergence_predictor_kernel(
    const double* prices,
    const double* rsi_values,
    double* output_predictions,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx >= 2 * period) {
            // Check for bullish divergence (price makes lower low, RSI makes higher low)
            bool price_lower_low = prices[idx] < prices[idx - period] && 
                                  prices[idx - period] < prices[idx - 2 * period];
            bool rsi_higher_low = rsi_values[idx] > rsi_values[idx - period] && 
                                 rsi_values[idx - period] > rsi_values[idx - 2 * period];

            if (price_lower_low && rsi_higher_low) {
                output_predictions[idx] = 1.0;  // Bullish divergence signal
            }
            // Check for bearish divergence (price makes higher high, RSI makes lower high)
            else if (prices[idx] > prices[idx - period] && 
                     prices[idx - period] > prices[idx - 2 * period] &&
                     rsi_values[idx] < rsi_values[idx - period] && 
                     rsi_values[idx - period] < rsi_values[idx - 2 * period]) {
                output_predictions[idx] = -1.0;  // Bearish divergence signal
            } else {
                output_predictions[idx] = 0.0;   // No signal
            }
        } else {
            output_predictions[idx] = 0.0;       // Not enough data
        }
    }
}

// Heuristic predictor kernel for support/resistance level detection
__global__ void sr_level_predictor_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output_predictions,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx >= period) {
            // Find local minima/maxima in the lookback period
            double local_high = high[idx];
            double local_low = low[idx];
            
            for (int i = 1; i <= period; i++) {
                if (idx >= i) {
                    if (high[idx - i] > local_high) local_high = high[idx - i];
                    if (low[idx - i] < local_low) local_low = low[idx - i];
                }
            }
            
            // Check if current price is near resistance (high) or support (low)
            double current_price = close[idx];
            double range = local_high - local_low;
            
            if (range > 0) {  // Avoid division by zero
                double position_in_range = (current_price - local_low) / range;
                
                // If price is near top of range (potential resistance bounce)
                if (position_in_range > 0.8) {
                    output_predictions[idx] = -0.5;  // Potential sell signal
                }
                // If price is near bottom of range (potential support bounce)
                else if (position_in_range < 0.2) {
                    output_predictions[idx] = 0.5;   // Potential buy signal
                }
                else {
                    output_predictions[idx] = 0.0;   // Neutral
                }
            } else {
                output_predictions[idx] = 0.0;       // No range to work with
            }
        } else {
            output_predictions[idx] = 0.0;           // Not enough data
        }
    }
}

// Heuristic predictor kernel for momentum reversal
__global__ void momentum_reversal_predictor_kernel(
    const double* prices,
    const double* rsi_values,
    double* output_predictions,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx >= period) {
            // Check if we have strong momentum followed by weakening
            bool was_overbought = rsi_values[idx - 1] > 70.0;
            bool was_oversold = rsi_values[idx - 1] < 30.0;
            bool is_reversing = (was_overbought && rsi_values[idx] < rsi_values[idx - 1]) ||
                               (was_oversold && rsi_values[idx] > rsi_values[idx - 1]);
            
            if (was_overbought && is_reversing) {
                output_predictions[idx] = -1.0;  // Potential reversal from overbought
            } else if (was_oversold && is_reversing) {
                output_predictions[idx] = 1.0;   // Potential reversal from oversold
            } else {
                output_predictions[idx] = 0.0;   // No clear reversal signal
            }
        } else {
            output_predictions[idx] = 0.0;       // Not enough data
        }
    }
}

// Batch predictor combiner - combines multiple heuristic predictions
__global__ void heuristic_combiner_kernel(
    const double* div_pred,
    const double* sr_pred,
    const double* mom_pred,
    double* combined_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Simple weighted combination of predictions
        double combined_score = 0.4 * div_pred[idx] + 0.3 * sr_pred[idx] + 0.3 * mom_pred[idx];
        
        // Clamp the result to [-1, 1] range
        if (combined_score > 1.0) combined_score = 1.0;
        if (combined_score < -1.0) combined_score = -1.0;
        
        combined_output[idx] = combined_score;
    }
}

// Kernel to convert heuristic predictions to signal format compatible with consensus kernel
__global__ void heuristic_to_signal_kernel(
    const double* heuristic_values,
    float* signal_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Convert heuristic values to signal format (0.0 to 1.0 range for buys, 0.0 to -1.0 for sells)
        if (heuristic_values[idx] > 0.0) {
            signal_output[idx] = fminf(1.0f, (float)heuristic_values[idx]);  // Buy signal
        } else {
            signal_output[idx] = fmaxf(0.0f, (float)heuristic_values[idx] + 1.0f);  // Sell signal (mapped to 0.0-1.0 range)
        }
    }
}