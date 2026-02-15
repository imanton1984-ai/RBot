#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <math.h>

// Series-based EMA kernel - each block processes one series
extern "C" __global__ void ema_series_kernel(
    const double* input, 
    double* output, 
    int n, 
    int period, 
    int batch
) {
    int s = blockIdx.x;  // series index
    if (s >= batch) return;
    
    const double* x = input + (size_t)s * n;
    double* y = output + (size_t)s * n;
    
    double multiplier = 2.0 / (period + 1.0);
    
    // Initialize first value
    if (n > 0) {
        y[0] = x[0];
    }
    
    // Calculate EMA sequentially for this series
    for (int i = 1; i < n; i++) {
        y[i] = (x[i] - y[i-1]) * multiplier + y[i-1];
    }
}

// Series-based RSI (Wilder smoothing) kernel
extern "C" __global__ void rsi_wilder_series_kernel(
    const double* close, 
    double* out,
    int n, 
    int period, 
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    const double* x = close + (size_t)s * n;
    double* y = out + (size_t)s * n;

    for (int i = 0; i < n; i++) y[i] = NAN;
    if (n < period + 1) return;

    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 1; i <= period; i++) {
        double ch = x[i] - x[i-1];
        if (ch > 0) avg_gain += ch; else avg_loss += -ch;
    }
    avg_gain /= (double)period;
    avg_loss /= (double)period;

    auto rsi_of = [&](double g, double l) {
        if (l == 0.0) return 100.0;
        double rs = g / l;
        return 100.0 - (100.0 / (1.0 + rs));
    };

    y[period] = rsi_of(avg_gain, avg_loss);

    for (int i = period + 1; i < n; i++) {
        double ch = x[i] - x[i-1];
        double gain = ch > 0 ? ch : 0.0;
        double loss = ch < 0 ? -ch : 0.0;
        avg_gain = (avg_gain * (period - 1.0) + gain) / period;
        avg_loss = (avg_loss * (period - 1.0) + loss) / period;
        y[i] = rsi_of(avg_gain, avg_loss);
    }
}

// Series-based MACD kernel
extern "C" __global__ void macd_series_kernel(
    const double* input,
    double* macd_line,
    double* signal_line,
    double* histogram,
    int n,
    int fast_period,
    int slow_period,
    int signal_period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* x = input + (size_t)s * n;
    double* macd_out = macd_line + (size_t)s * n;
    double* sig_out = signal_line + (size_t)s * n;
    double* hist_out = histogram + (size_t)s * n;

    // Calculate EMA multipliers
    double fast_mult = 2.0 / (fast_period + 1.0);
    double slow_mult = 2.0 / (slow_period + 1.0);
    double sig_mult = 2.0 / (signal_period + 1.0);

    // Initialize arrays
    for (int i = 0; i < n; i++) {
        macd_out[i] = NAN;
        sig_out[i] = NAN;
        hist_out[i] = NAN;
    }

    // Calculate fast and slow EMAs sequentially
    double fast_ema = x[0];
    double slow_ema = x[0];
    
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fast_ema = (x[i] - fast_ema) * fast_mult + fast_ema;
            slow_ema = (x[i] - slow_ema) * slow_mult + slow_ema;
        } else {
            fast_ema = x[0];
            slow_ema = x[0];
        }
        
        macd_out[i] = fast_ema - slow_ema;
    }

    // Calculate signal line (EMA of MACD line)
    double signal_ema = macd_out[0];
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            signal_ema = (macd_out[i] - signal_ema) * sig_mult + signal_ema;
        } else {
            signal_ema = macd_out[0];
        }
        sig_out[i] = signal_ema;
        hist_out[i] = macd_out[i] - sig_out[i];
    }
}

// Series-based ATR (Average True Range) with Wilder smoothing
extern "C" __global__ void atr_wilder_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* out = output + (size_t)s * n;

    for (int i = 0; i < n; i++) out[i] = NAN;
    if (n < period) return;

    // Calculate True Range for each bar
    double* tr = new double[n];
    for (int i = 1; i < n; i++) {
        double hl = h[i] - l[i];
        double hc = fabs(h[i] - c[i-1]);
        double lc = fabs(l[i] - c[i-1]);
        tr[i] = fmax(hl, fmax(hc, lc));
    }

    // Calculate initial ATR (simple average of first 'period' TR values)
    double sum_tr = 0.0;
    for (int i = 1; i <= period; i++) {
        sum_tr += tr[i];
    }
    out[period] = sum_tr / period;

    // Calculate subsequent ATR values using Wilder's smoothing
    for (int i = period + 1; i < n; i++) {
        out[i] = (out[i-1] * (period - 1) + tr[i]) / period;
    }

    delete[] tr;
}

// Series-based ADX (Average Directional Index) with Wilder smoothing
extern "C" __global__ void adx_wilder_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* out = output + (size_t)s * n;

    for (int i = 0; i < n; i++) out[i] = NAN;
    if (n < period * 2) return; // Need at least 2*period for ADX

    // Calculate True Range, +DM, -DM
    double* tr = new double[n];
    double* plus_dm = new double[n];
    double* minus_dm = new double[n];
    
    for (int i = 1; i < n; i++) {
        double up_move = h[i] - h[i-1];
        double down_move = l[i-1] - l[i];
        double hl = h[i] - l[i];
        double hc = fabs(h[i] - c[i-1]);
        double lc = fabs(l[i] - c[i-1]);
        
        tr[i] = fmax(hl, fmax(hc, lc));
        
        plus_dm[i] = (up_move > down_move && up_move > 0) ? up_move : 0;
        minus_dm[i] = (down_move > up_move && down_move > 0) ? down_move : 0;
    }

    // Smooth TR, +DM, -DM using Wilder's method
    double* atr = new double[n];
    double* plus_di = new double[n];
    double* minus_di = new double[n];
    
    // Initialize smoothed values
    double sum_tr = 0, sum_plus = 0, sum_minus = 0;
    for (int i = 1; i <= period; i++) {
        sum_tr += tr[i];
        sum_plus += plus_dm[i];
        sum_minus += minus_dm[i];
    }
    atr[period] = sum_tr;
    plus_di[period] = (atr[period] != 0) ? (sum_plus / atr[period]) * 100 : 0;
    minus_di[period] = (atr[period] != 0) ? (sum_minus / atr[period]) * 100 : 0;

    // Continue smoothing
    for (int i = period + 1; i < n; i++) {
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i];
        plus_di[i] = plus_di[i-1] - (plus_di[i-1] / period) + (plus_dm[i] / atr[i]);
        minus_di[i] = minus_di[i-1] - (minus_di[i-1] / period) + (minus_dm[i] / atr[i]);
    }

    // Calculate DX and ADX
    double* dx = new double[n];
    for (int i = period; i < n; i++) {
        double sum_di = fabs(plus_di[i]) + fabs(minus_di[i]);
        dx[i] = (sum_di != 0) ? (fabs(plus_di[i] - minus_di[i]) / sum_di) * 100 : 0;
    }

    // Initialize ADX
    double sum_dx = 0;
    for (int i = period; i < period * 2; i++) {
        sum_dx += dx[i];
    }
    out[period * 2] = sum_dx / period;

    // Continue ADX calculation
    for (int i = period * 2 + 1; i < n; i++) {
        out[i] = (out[i-1] * (period - 1) + dx[i]) / period;
    }

    delete[] tr;
    delete[] plus_dm;
    delete[] minus_dm;
    delete[] atr;
    delete[] plus_di;
    delete[] minus_di;
    delete[] dx;
}

// Series-based Bollinger Bands kernel
extern "C" __global__ void bb_series_kernel(
    const double* input,
    double* upper_band,
    double* middle_band,
    double* lower_band,
    int n,
    int period,
    double num_std_dev,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* x = input + (size_t)s * n;
    double* upper = upper_band + (size_t)s * n;
    double* mid = middle_band + (size_t)s * n;
    double* lower = lower_band + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        upper[i] = NAN;
        mid[i] = NAN;
        lower[i] = NAN;
    }

    if (n < period) return;

    // Calculate SMA and standard deviation for each position
    for (int i = period - 1; i < n; i++) {
        // Calculate SMA (middle band)
        double sum = 0.0;
        for (int j = 0; j < period; j++) {
            sum += x[i - j];
        }
        mid[i] = sum / period;

        // Calculate standard deviation
        double sum_sq_diff = 0.0;
        for (int j = 0; j < period; j++) {
            double diff = x[i - j] - mid[i];
            sum_sq_diff += diff * diff;
        }
        double std_dev = sqrt(sum_sq_diff / period);

        // Upper and lower bands
        upper[i] = mid[i] + (num_std_dev * std_dev);
        lower[i] = mid[i] - (num_std_dev * std_dev);
    }
}

// Series-based SMA kernel
extern "C" __global__ void sma_series_kernel(
    const double* input, 
    double* output, 
    int n, 
    int period, 
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* x = input + (size_t)s * n;
    double* y = output + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        y[i] = NAN;
    }

    if (n < period) return;

    // Calculate SMA for each position
    for (int i = period - 1; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < period; j++) {
            sum += x[i - j];
        }
        y[i] = sum / period;
    }
}

// Series-based OBV kernel
extern "C" __global__ void obv_series_kernel(
    const double* close_prices,
    const double* volumes,
    double* obv,
    int n,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* close = close_prices + (size_t)s * n;
    const double* vol = volumes + (size_t)s * n;
    double* out = obv + (size_t)s * n;

    if (n == 0) return;

    out[0] = vol[0];

    for (int i = 1; i < n; i++) {
        if (close[i] > close[i-1]) {
            out[i] = out[i-1] + vol[i];
        } else if (close[i] < close[i-1]) {
            out[i] = out[i-1] - vol[i];
        } else {
            out[i] = out[i-1];
        }
    }
}

// Series-based VWAP kernel
extern "C" __global__ void vwap_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    double* output,
    int n,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    const double* v = volume + (size_t)s * n;
    double* out = output + (size_t)s * n;

    if (n == 0) return;

    double cum_price_vol = 0.0;
    double cum_vol = 0.0;

    for (int i = 0; i < n; i++) {
        double typical_price = (h[i] + l[i] + c[i]) / 3.0;
        cum_price_vol += typical_price * v[i];
        cum_vol += v[i];
        out[i] = (cum_vol != 0.0) ? cum_price_vol / cum_vol : 0.0;
    }
}

// Series-based Stochastic Oscillator kernel
extern "C" __global__ void stoch_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* k_values,
    double* d_values,
    int n,
    int k_period,
    int d_period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* k_out = k_values + (size_t)s * n;
    double* d_out = d_values + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        k_out[i] = NAN;
        d_out[i] = NAN;
    }

    if (n < k_period) return;

    // Calculate %K for each position
    for (int i = k_period - 1; i < n; i++) {
        double highest_high = h[i];
        double lowest_low = l[i];

        for (int j = 0; j < k_period; j++) {
            if (i >= j) {
                if (h[i - j] > highest_high) {
                    highest_high = h[i - j];
                }
                if (l[i - j] < lowest_low) {
                    lowest_low = l[i - j];
                }
            }
        }

        // Calculate %K
        if (highest_high != lowest_low) {
            k_out[i] = ((c[i] - lowest_low) / (highest_high - lowest_low)) * 100.0;
        } else {
            k_out[i] = 50.0; // Neutral value when high equals low
        }
    }

    // Calculate %D (moving average of %K)
    if (n >= k_period + d_period - 1) {
        for (int i = k_period + d_period - 2; i < n; i++) {
            double sum_k = 0.0;
            for (int j = 0; j < d_period; j++) {
                if (i >= j) {
                    sum_k += k_out[i - j];
                }
            }
            d_out[i] = sum_k / d_period;
        }
    }
}

// Series-based Williams %R kernel
extern "C" __global__ void williams_r_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* out = output + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        out[i] = NAN;
    }

    if (n < period) return;

    // Calculate Williams %R for each position
    for (int i = period - 1; i < n; i++) {
        double highest_high = h[i];
        double lowest_low = l[i];

        for (int j = 0; j < period; j++) {
            if (i >= j) {
                if (h[i - j] > highest_high) {
                    highest_high = h[i - j];
                }
                if (l[i - j] < lowest_low) {
                    lowest_low = l[i - j];
                }
            }
        }

        // Calculate Williams %R
        if (highest_high != lowest_low) {
            out[i] = ((highest_high - c[i]) / (highest_high - lowest_low)) * -100.0;
        } else {
            out[i] = -50.0; // Neutral value when high equals low
        }
    }
}

// Series-based CCI (Commodity Channel Index) kernel
extern "C" __global__ void cci_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* output,
    int n,
    int period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* out = output + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        out[i] = NAN;
    }

    if (n < period) return;

    // Calculate CCI for each position
    for (int i = period - 1; i < n; i++) {
        // Calculate Typical Price (TP)
        double tp = (h[i] + l[i] + c[i]) / 3.0;

        // Calculate Simple Moving Average of TP
        double sma_tp = 0.0;
        for (int j = 0; j < period; j++) {
            sma_tp += (h[i - j] + l[i - j] + c[i - j]) / 3.0;
        }
        sma_tp /= period;

        // Calculate Mean Deviation
        double mean_dev = 0.0;
        for (int j = 0; j < period; j++) {
            double tp_val = (h[i - j] + l[i - j] + c[i - j]) / 3.0;
            mean_dev += fabs(tp_val - sma_tp);
        }
        mean_dev /= period;

        // Calculate CCI
        if (mean_dev != 0.0) {
            out[i] = (tp - sma_tp) / (0.015 * mean_dev);
        } else {
            out[i] = 0.0;
        }
    }
}

// Series-based Alligator kernel
extern "C" __global__ void alligator_series_kernel(
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
    int lips_offset,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* src = source + (size_t)s * n;
    double* jaw_out = jaw + (size_t)s * n;
    double* teeth_out = teeth + (size_t)s * n;
    double* lips_out = lips + (size_t)s * n;

    for (int i = 0; i < n; i++) {
        jaw_out[i] = NAN;
        teeth_out[i] = NAN;
        lips_out[i] = NAN;
    }

    if (n < jaw_period || n < teeth_period || n < lips_period) return;

    // Calculate SMMA for Jaw (blue line)
    for (int i = jaw_period - 1; i < n; i++) {
        if (i == jaw_period - 1) {
            // Calculate simple moving average for the first value
            double sum = 0.0;
            for (int j = 0; j < jaw_period; j++) {
                sum += src[i - j];
            }
            jaw_out[i] = sum / jaw_period;
        } else {
            // Calculate subsequent SMMA values
            jaw_out[i] = (jaw_out[i-1] * (jaw_period - 1) + src[i]) / jaw_period;
        }
    }

    // Calculate SMMA for Teeth (red line)
    for (int i = teeth_period - 1; i < n; i++) {
        if (i == teeth_period - 1) {
            // Calculate simple moving average for the first value
            double sum = 0.0;
            for (int j = 0; j < teeth_period; j++) {
                sum += src[i - j];
            }
            teeth_out[i] = sum / teeth_period;
        } else {
            // Calculate subsequent SMMA values
            teeth_out[i] = (teeth_out[i-1] * (teeth_period - 1) + src[i]) / teeth_period;
        }
    }

    // Calculate SMMA for Lips (green line)
    for (int i = lips_period - 1; i < n; i++) {
        if (i == lips_period - 1) {
            // Calculate simple moving average for the first value
            double sum = 0.0;
            for (int j = 0; j < lips_period; j++) {
                sum += src[i - j];
            }
            lips_out[i] = sum / lips_period;
        } else {
            // Calculate subsequent SMMA values
            lips_out[i] = (lips_out[i-1] * (lips_period - 1) + src[i]) / lips_period;
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

// Batch version of Bollinger Bands kernel
__global__ void bb_batch_kernel(const double* input, double* upper_band, double* middle_band, double* lower_band, int n, int period, double num_std_dev) {
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

// Batch version of OBV kernel
__global__ void obv_batch_kernel(const double* close_prices, const double* volumes, double* obv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx == 0) {
            obv[idx] = volumes[idx];
        } else {
            if (close_prices[idx] > close_prices[idx - 1]) {
                obv[idx] = obv[idx - 1] + volumes[idx];
            } else if (close_prices[idx] < close_prices[idx - 1]) {
                obv[idx] = obv[idx - 1] - volumes[idx];
            } else {
                obv[idx] = obv[idx - 1];
            }
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

            // For simplicity in this kernel, we'll use a simplified approach
            // In practice, you'd need to properly calculate smoothed TR and DI values
            // and then derive the DX and ADX

            // Placeholder for ADX calculation
            output[idx] = tr; // This is a simplified placeholder
        }
    }
}

// Batch version of ADX kernel
__global__ void adx_batch_kernel(
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
                    if (i > 0) {
                        double h_minus_l_temp = high[i] - low[i];
                        double h_minus_c_temp = fabs(high[i] - close[i - 1]);
                        double l_minus_c_temp = fabs(low[i] - close[i - 1]);
                        double tr_temp = fmax(h_minus_l_temp, fmax(h_minus_c_temp, l_minus_c_temp));
                        sum_tr += tr_temp;
                    }
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

// Batch version of ATR kernel
__global__ void atr_batch_kernel(
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
                    if (i > 0) {
                        double h_m_l = high[i] - low[i];
                        double h_m_c = fabs(high[i] - close[i - 1]);
                        double l_m_c = fabs(low[i] - close[i - 1]);
                        double tr_temp = fmax(h_m_l, fmax(h_m_c, l_m_c));
                        sum_tr += tr_temp;
                    }
                }
                output[idx] = sum_tr / period;
            } else {
                // Apply Wilder's smoothing formula: ATR = [(Period-1) * ATR_prev + TR] / Period
                output[idx] = (output[idx - 1] * (period - 1) + tr) / period;
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

// Batch version of CCI kernel
__global__ void cci_batch_kernel(
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

// Batch version of Stochastic Oscillator kernel
__global__ void stochastic_batch_kernel(
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

// Batch version of VWAP kernel
__global__ void vwap_batch_kernel(
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

// Batch version of Williams %R kernel
__global__ void williams_r_batch_kernel(
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

// Batch version of Alligator indicator kernel
__global__ void alligator_batch_kernel(
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