#include <cuda_runtime.h>
#include <stdio.h>

// Simple moving average kernel
__global__ void sma_kernel(const double* input, double* output, int n, int period) {
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

// Exponential moving average kernel
__global__ void ema_kernel(const double* input, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        output[idx] = input[idx];
    } else if (idx < n) {
        double multiplier = 2.0 / (period + 1.0);
        if (idx == 1) {
            output[idx] = input[0]; // Use previous EMA value
        } else {
            output[idx] = (input[idx] - output[idx - 1]) * multiplier + output[idx - 1];
        }
    }
}

// RSI (Relative Strength Index) kernel
__global__ void rsi_kernel(const double* input, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx < period) {
            output[idx] = NAN;
        } else {
            // Calculate initial average gain and loss
            double avg_gain = 0.0;
            double avg_loss = 0.0;
            
            for (int i = 1; i <= period; i++) {
                double change = input[idx - i + 1] - input[idx - i];
                if (change > 0.0) {
                    avg_gain += change;
                } else {
                    avg_loss += fabs(change);
                }
            }
            
            avg_gain /= period;
            avg_loss /= period;
            
            // Calculate RSI
            double rs = (avg_loss != 0.0) ? avg_gain / avg_loss : 0.0;
            output[idx] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
}

// MACD kernel - calculates MACD line, signal line, and histogram
__global__ void macd_kernel(
    const double* input, 
    double* macd_line, 
    double* signal_line, 
    double* histogram, 
    int n, 
    int fast_period, 
    int slow_period, 
    int signal_period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Calculate EMA for fast and slow periods
        double multiplier_fast = 2.0 / (fast_period + 1.0);
        double multiplier_slow = 2.0 / (slow_period + 1.0);
        double multiplier_signal = 2.0 / (signal_period + 1.0);
        
        // For simplicity, we'll use a simplified approach
        // In practice, you'd need to properly calculate EMAs
        if (idx >= slow_period) {
            // Calculate fast EMA approximation
            double fast_ema = input[idx];
            if (idx > 0) {
                fast_ema = (input[idx] - input[idx-1]) * multiplier_fast + input[idx-1];
            }
            
            // Calculate slow EMA approximation
            double slow_ema = input[idx];
            if (idx > 0) {
                slow_ema = (input[idx] - input[idx-1]) * multiplier_slow + input[idx-1];
            }
            
            // MACD line
            macd_line[idx] = fast_ema - slow_ema;
            
            // Signal line (EMA of MACD line)
            if (idx >= slow_period + signal_period) {
                signal_line[idx] = (macd_line[idx] - macd_line[idx-1]) * multiplier_signal + macd_line[idx-1];
                
                // Histogram
                histogram[idx] = macd_line[idx] - signal_line[idx];
            } else {
                signal_line[idx] = NAN;
                histogram[idx] = NAN;
            }
        } else {
            macd_line[idx] = NAN;
            signal_line[idx] = NAN;
            histogram[idx] = NAN;
        }
    }
}

// Bollinger Bands kernel
__global__ void bb_kernel(const double* input, double* upper_band, double* middle_band, double* lower_band, int n, int period, double num_std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx < period - 1) {
            upper_band[idx] = NAN;
            middle_band[idx] = NAN;
            lower_band[idx] = NAN;
        } else {
            // Calculate SMA (middle band)
            double sum = 0.0;
            for (int i = 0; i < period; i++) {
                sum += input[idx - i];
            }
            middle_band[idx] = sum / period;
            
            // Calculate standard deviation
            double sum_sq_diff = 0.0;
            for (int i = 0; i < period; i++) {
                double diff = input[idx - i] - middle_band[idx];
                sum_sq_diff += diff * diff;
            }
            double std_dev = sqrt(sum_sq_diff / period);
            
            // Upper and lower bands
            upper_band[idx] = middle_band[idx] + (num_std_dev * std_dev);
            lower_band[idx] = middle_band[idx] - (num_std_dev * std_dev);
        }
    }
}

// On Balance Volume (OBV) kernel
__global__ void obv_kernel(const double* close_prices, const double* volumes, double* obv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0 && idx < n) {
        obv[idx] = volumes[idx];
    } else if (idx < n) {
        if (close_prices[idx] > close_prices[idx - 1]) {
            obv[idx] = obv[idx - 1] + volumes[idx];
        } else if (close_prices[idx] < close_prices[idx - 1]) {
            obv[idx] = obv[idx - 1] - volumes[idx];
        } else {
            obv[idx] = obv[idx - 1];
        }
    }
}

// Average Directional Index (ADX) kernel
__global__ void adx_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period + 1) {
            output[idx] = NAN;
        } else {
            // Calculate True Range (TR)
            double tr = 0.0;
            if (idx > 0) {
                double h_minus_l = high[idx] - low[idx];
                double h_minus_c = fabs(high[idx] - close[idx - 1]);
                double l_minus_c = fabs(low[idx] - close[idx - 1]);
                tr = fmax(h_minus_l, fmax(h_minus_c, l_minus_c));
            }

            // Calculate Directional Movement (+DM and -DM)
            double plus_dm = 0.0;
            double minus_dm = 0.0;
            if (idx > 0) {
                double up_move = high[idx] - high[idx - 1];
                double down_move = low[idx - 1] - low[idx];

                plus_dm = (up_move > down_move && up_move > 0.0) ? up_move : 0.0;
                minus_dm = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
            }

            // For simplicity in this kernel, we'll use a simplified approach
            // In practice, you'd need to properly calculate smoothed TR and DI values
            // and then derive the DX and ADX

            // Placeholder for ADX calculation
            output[idx] = tr; // This is a simplified placeholder
        }
    }
}

// Average True Range (ATR) kernel
__global__ void atr_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period) {
            output[idx] = NAN;
        } else {
            // Calculate True Range (TR)
            double tr = 0.0;
            if (idx > 0) {
                double h_minus_l = high[idx] - low[idx];
                double h_minus_c = fabs(high[idx] - close[idx - 1]);
                double l_minus_c = fabs(low[idx] - close[idx - 1]);
                tr = fmax(h_minus_l, fmax(h_minus_c, l_minus_c));
            }

            // For the first ATR value, calculate simple average
            if (idx == period) {
                double sum_tr = 0.0;
                for (int i = 1; i <= period; i++) {
                    double h_minus_l_temp = high[i] - low[i];
                    double h_minus_c_temp = fabs(high[i] - close[i - 1]);
                    double l_minus_c_temp = fabs(low[i] - close[i - 1]);
                    double tr_temp = fmax(h_minus_l_temp, fmax(h_minus_c_temp, l_minus_c_temp));
                    sum_tr += tr_temp;
                }
                output[idx] = sum_tr / period;
            } else if (idx > period) {
                // Calculate subsequent ATR values using Wilder's smoothing
                output[idx] = (output[idx - 1] * (period - 1) + tr) / period;
            } else {
                output[idx] = tr;
            }
        }
    }
}

// Commodity Channel Index (CCI) kernel
__global__ void cci_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period) {
            output[idx] = NAN;
        } else {
            // Calculate Typical Price (TP)
            double tp = (high[idx] + low[idx] + close[idx]) / 3.0;

            // Calculate Simple Moving Average of TP
            double sma_tp = 0.0;
            for (int i = 0; i < period; i++) {
                sma_tp += (high[idx - i] + low[idx - i] + close[idx - i]) / 3.0;
            }
            sma_tp /= period;

            // Calculate Mean Deviation
            double mean_dev = 0.0;
            for (int i = 0; i < period; i++) {
                double tp_val = (high[idx - i] + low[idx - i] + close[idx - i]) / 3.0;
                mean_dev += fabs(tp_val - sma_tp);
            }
            mean_dev /= period;

            // Calculate CCI
            if (mean_dev != 0.0) {
                output[idx] = (tp - sma_tp) / (0.015 * mean_dev);
            } else {
                output[idx] = 0.0;
            }
        }
    }
}

// Stochastic Oscillator kernel
__global__ void stochastic_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* k_values,
    double* d_values,
    int n,
    int k_period,
    int d_period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < k_period - 1) {
            k_values[idx] = NAN;
        } else {
            // Find highest high and lowest low in the k_period
            double highest_high = high[idx];
            double lowest_low = low[idx];

            for (int i = 0; i < k_period; i++) {
                if (idx >= i) {
                    if (high[idx - i] > highest_high) {
                        highest_high = high[idx - i];
                    }
                    if (low[idx - i] < lowest_low) {
                        lowest_low = low[idx - i];
                    }
                }
            }

            // Calculate %K
            if (highest_high != lowest_low) {
                k_values[idx] = ((close[idx] - lowest_low) / (highest_high - lowest_low)) * 100.0;
            } else {
                k_values[idx] = 50.0; // Neutral value when high equals low
            }
        }

        // Calculate %D (moving average of %K)
        if (idx >= k_period + d_period - 2) {
            double sum_k = 0.0;
            for (int i = 0; i < d_period; i++) {
                if (idx >= i) {
                    sum_k += k_values[idx - i];
                }
            }
            d_values[idx] = sum_k / d_period;
        } else {
            d_values[idx] = NAN;
        }
    }
}

// VWAP (Volume Weighted Average Price) kernel
__global__ void vwap_kernel(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    double* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx == 0) {
            double typical_price = (high[idx] + low[idx] + close[idx]) / 3.0;
            output[idx] = typical_price * volume[idx];
        } else {
            double typical_price = (high[idx] + low[idx] + close[idx]) / 3.0;
            output[idx] = output[idx - 1] + (typical_price * volume[idx]);
        }
    }
}

// Williams %R kernel
__global__ void williams_r_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx < period) {
            output[idx] = NAN;
        } else {
            // Find highest high and lowest low in the period
            double highest_high = high[idx];
            double lowest_low = low[idx];

            for (int i = 0; i < period; i++) {
                if (idx >= i) {
                    if (high[idx - i] > highest_high) {
                        highest_high = high[idx - i];
                    }
                    if (low[idx - i] < lowest_low) {
                        lowest_low = low[idx - i];
                    }
                }
            }

            // Calculate Williams %R
            if (highest_high != lowest_low) {
                output[idx] = ((highest_high - close[idx]) / (highest_high - lowest_low)) * -100.0;
            } else {
                output[idx] = -50.0; // Neutral value when high equals low
            }
        }
    }
}

// Alligator indicator kernel (uses SMMA internally)
__global__ void alligator_kernel(
    const double* source,
    double* jaw,
    double* teeth,
    double* lips,
    int n,
    int jaw_period,
    int teeth_period,
    int lips_period,
    int jaw_offset,
    int teeth_offset,
    int lips_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate SMMA for Jaw (blue line)
    if (idx < n) {
        if (idx < jaw_period) {
            jaw[idx] = NAN;
        } else {
            if (idx == jaw_period - 1) {
                // Calculate simple moving average for the first value
                double sum = 0.0;
                for (int i = 0; i < jaw_period; i++) {
                    sum += source[idx - i];
                }
                jaw[idx] = sum / jaw_period;
            } else {
                // Calculate subsequent SMMA values
                jaw[idx] = (jaw[idx - 1] * (jaw_period - 1) + source[idx]) / jaw_period;
            }
        }

        // Calculate SMMA for Teeth (red line)
        if (idx < teeth_period) {
            teeth[idx] = NAN;
        } else {
            if (idx == teeth_period - 1) {
                // Calculate simple moving average for the first value
                double sum = 0.0;
                for (int i = 0; i < teeth_period; i++) {
                    sum += source[idx - i];
                }
                teeth[idx] = sum / teeth_period;
            } else {
                // Calculate subsequent SMMA values
                teeth[idx] = (teeth[idx - 1] * (teeth_period - 1) + source[idx]) / teeth_period;
            }
        }

        // Calculate SMMA for Lips (green line)
        if (idx < lips_period) {
            lips[idx] = NAN;
        } else {
            if (idx == lips_period - 1) {
                // Calculate simple moving average for the first value
                double sum = 0.0;
                for (int i = 0; i < lips_period; i++) {
                    sum += source[idx - i];
                }
                lips[idx] = sum / lips_period;
            } else {
                // Calculate subsequent SMMA values
                lips[idx] = (lips[idx - 1] * (lips_period - 1) + source[idx]) / lips_period;
            }
        }
    }
}