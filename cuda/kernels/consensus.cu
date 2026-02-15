#include <cuda_runtime.h>
#include <stdio.h>

// Consensus kernel that combines signals from multiple sources (ML models and heuristic indicators)
// Input: Multiple signal arrays from ML models and heuristic indicators
// Output: Final trading signals (0=Hold, 1=Buy, 2=Sell)
extern "C" __global__ void final_consensus_kernel(
    const float* ml_1,           // ML model 1 predictions
    const float* ml_2,           // ML model 2 predictions  
    const float* heur_1,         // Heuristic indicator 1
    const float* heur_2,         // Heuristic indicator 2
    uint8_t* final_signals,      // Output: 0=Hold, 1=Buy, 2=Sell
    int n                        // Number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Initialize signal strength
        float buy_strength = 0.0f;
        float sell_strength = 0.0f;

        // Process ML model 1 (probability of up move)
        if (ml_1[idx] > 0.6f) {
            buy_strength += ml_1[idx];
        } else if (ml_1[idx] < 0.4f) {
            sell_strength += (1.0f - ml_1[idx]);
        }

        // Process ML model 2 (probability of up move)
        if (ml_2[idx] > 0.6f) {
            buy_strength += ml_2[idx];
        } else if (ml_2[idx] < 0.4f) {
            sell_strength += (1.0f - ml_2[idx]);
        }

        // Process heuristic indicator 1 (1=buy signal, -1=sell signal, 0=no signal)
        if (heur_1[idx] > 0.5f) {
            buy_strength += heur_1[idx];
        } else if (heur_1[idx] < -0.5f) {
            sell_strength += abs(heur_1[idx]);
        }

        // Process heuristic indicator 2 (1=buy signal, -1=sell signal, 0=no signal)
        if (heur_2[idx] > 0.5f) {
            buy_strength += heur_2[idx];
        } else if (heur_2[idx] < -0.5f) {
            sell_strength += abs(heur_2[idx]);
        }

        // Determine final signal based on combined strengths
        if (buy_strength > sell_strength && buy_strength > 0.5f) {
            final_signals[idx] = 1;  // BUY
        } else if (sell_strength > buy_strength && sell_strength > 0.5f) {
            final_signals[idx] = 2;  // SELL
        } else {
            final_signals[idx] = 0;  // HOLD
        }
    }
}

// Kernel for heuristic signal generation based on RSI divergence
extern "C" __global__ void check_rsi_divergence_kernel(
    const double* prices,
    const double* rsi_values,
    double* output_signals,
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
                output_signals[idx] = 1.0;  // Bullish divergence signal
            }
            // Check for bearish divergence (price makes higher high, RSI makes lower high)
            else if (prices[idx] > prices[idx - period] && 
                     prices[idx - period] > prices[idx - 2 * period] &&
                     rsi_values[idx] < rsi_values[idx - period] && 
                     rsi_values[idx - period] < rsi_values[idx - 2 * period]) {
                output_signals[idx] = -1.0;  // Bearish divergence signal
            } else {
                output_signals[idx] = 0.0;   // No signal
            }
        } else {
            output_signals[idx] = 0.0;       // Not enough data
        }
    }
}

// Enhanced RSI kernel for batch processing
extern "C" __global__ void rsi_batch_kernel(const double* input, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period) {
            output[idx] = NAN;
        } else {
            // Calculate initial average gain and loss using a sliding window
            double avg_gain = 0.0;
            double avg_loss = 0.0;
            int count = 0;

            // Calculate initial averages for the first complete period
            for (int i = idx - period + 1; i <= idx; i++) {
                if (i > 0) {  // Ensure we don't go out of bounds
                    double change = input[i] - input[i - 1];
                    if (change > 0.0) {
                        avg_gain += change;
                    } else {
                        avg_loss += fabs(change);
                    }
                    count++;
                }
            }

            if (count > 0) {
                avg_gain /= count;
                avg_loss /= count;

                // Calculate RSI
                double rs = (avg_loss != 0.0) ? avg_gain / avg_loss : 0.0;
                output[idx] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                output[idx] = 50.0;  // Neutral value if no changes
            }
        }
    }
}

// Enhanced SMA kernel for batch processing
extern "C" __global__ void sma_batch_kernel(const double* input, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period - 1) {
            output[idx] = NAN;
        } else {
            double sum = 0.0;
            for (int i = 0; i < period; i++) {
                sum += input[idx - i];
            }
            output[idx] = sum / period;
        }
    }
}

// Enhanced EMA kernel for batch processing
extern "C" __global__ void ema_batch_kernel(const double* input, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        double multiplier = 2.0 / (period + 1.0);
        
        if (idx == 0) {
            output[idx] = input[idx];
        } else {
            // For the first few elements, we use a simplified approach
            // In production, you'd want to initialize with actual historical data
            if (idx < period) {
                // Use SMA for the first values as initialization
                double sum = 0.0;
                for (int i = 0; i <= idx; i++) {
                    sum += input[i];
                }
                if (idx == 0) {
                    output[idx] = input[idx];
                } else {
                    output[idx] = sum / (idx + 1);
                }
            } else {
                // Standard EMA calculation
                output[idx] = (input[idx] - output[idx - 1]) * multiplier + output[idx - 1];
            }
        }
    }
}

// Enhanced ATR kernel for batch processing
extern "C" __global__ void atr_batch_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx == 0) {
            output[idx] = 0.0;  // No ATR for first element
        } else if (idx < period) {
            // Calculate True Range for this period
            double h_minus_l = high[idx] - low[idx];
            double h_minus_c = fabs(high[idx] - close[idx - 1]);
            double l_minus_c = fabs(low[idx] - close[idx - 1]);
            double tr = fmax(h_minus_l, fmax(h_minus_c, l_minus_c));
            
            // For early periods, we just return the TR value
            output[idx] = tr;
        } else {
            // Calculate True Range
            double h_minus_l = high[idx] - low[idx];
            double h_minus_c = fabs(high[idx] - close[idx - 1]);
            double l_minus_c = fabs(low[idx] - close[idx - 1]);
            double tr = fmax(h_minus_l, fmax(h_minus_c, l_minus_c));

            // Calculate ATR using Wilder's smoothing
            if (idx == period) {
                // Calculate initial ATR as simple average of first period TRs
                double sum_tr = 0.0;
                for (int i = 1; i <= period; i++) {
                    double h_m_l = high[i] - low[i];
                    double h_m_c = fabs(high[i] - close[i - 1]);
                    double l_m_c = fabs(low[i] - close[i - 1]);
                    double tr_temp = fmax(h_m_l, fmax(h_m_c, l_m_c));
                    sum_tr += tr_temp;
                }
                output[idx] = sum_tr / period;
            } else {
                // Apply Wilder's smoothing formula: ATR = [(Period-1) * ATR_prev + TR] / Period
                output[idx] = (output[idx - 1] * (period - 1) + tr) / period;
            }
        }
    }
}

// Raw signal combination kernel - combines multiple indicator signals into feature set
extern "C" __global__ void raw_signals_combiner_kernel(
    const double* rsi_values,
    const double* sma_values,
    const double* ema_values,
    const double* atr_values,
    const double* bb_upper,
    const double* bb_lower,
    const double* bb_mid,
    double* feature_output,  // Combined feature array [n, num_features]
    int n,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Calculate position in the flattened feature matrix
        int base_idx = idx * num_features;
        
        // Store normalized features
        feature_output[base_idx + 0] = (rsi_values[idx] - 50.0) / 50.0;  // Normalize RSI to [-1, 1]
        feature_output[base_idx + 1] = sma_values[idx];
        feature_output[base_idx + 2] = ema_values[idx];
        feature_output[base_idx + 3] = atr_values[idx];
        
        // Bollinger Band features
        if (bb_mid[idx] != 0.0) {
            feature_output[base_idx + 4] = (bb_upper[idx] - bb_mid[idx]) / bb_mid[idx];  // BB width upper
            feature_output[base_idx + 5] = (bb_lower[idx] - bb_mid[idx]) / bb_mid[idx];  // BB width lower
            feature_output[base_idx + 6] = (bb_mid[idx] - bb_mid[idx > 0 ? idx - 1 : idx]) / bb_mid[idx > 0 ? idx - 1 : idx];  // BB slope
        } else {
            feature_output[base_idx + 4] = 0.0;
            feature_output[base_idx + 5] = 0.0;
            feature_output[base_idx + 6] = 0.0;
        }
    }
}