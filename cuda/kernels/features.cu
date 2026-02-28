#include <cuda_runtime.h>
#include <stdio.h>

// Kernel to combine features into a single matrix for XGBoost
// Accepts an array of pointers to feature arrays
extern "C" __global__ void combine_features_v1_kernel(
    const unsigned long long* feature_ptrs, // Array of pointers to feature arrays
    float* out, 
    int n, 
    int batch,
    int num_features // Number of features (should be 26)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch;
    
    if (idx < total_elements) {
        int row = idx;    // row index (global index of element in feature)
        int base = row * num_features;  // offset in output matrix
        
        // Iterate through all features and collect them into the output row
        for (int f = 0; f < num_features; f++) {
            // feature_ptrs[f] gives address of f-th feature array
            // Cast u64 back to float pointer and get the value at row index
            const float* feature_array = (const float*)feature_ptrs[f];
            out[base + f] = feature_array[row];
        }
    }
}

// Cast kernel: double to float
extern "C" __global__ void cast_f64_to_f32_kernel(const double* in, float* out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total) {
        out[idx] = (float)in[idx];
    }
}

// Fill constant kernel
extern "C" __global__ void fill_const_f32_kernel(float* out, float v, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total) {
        out[idx] = v;
    }
}

// Additional utility kernels for feature processing
// Kernel to calculate volume spike ratio
extern "C" __global__ void calculate_volume_spike_kernel(
    const double* volume,
    double* output,
    int n,
    int batch,
    int period
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* vol = volume + (size_t)s * n;
    double* out = output + (size_t)s * n;
    
    for (int i = 0; i < n; i++) {
        out[i] = 1.0;  // Default value
        
        if (i >= period) {
            // Calculate average volume over the period
            double sum = 0.0;
            for (int j = 1; j <= period; j++) {
                sum += vol[i - j];
            }
            double avg_vol = sum / period;
            
            // Calculate spike ratio
            if (avg_vol > 0) {
                out[i] = vol[i] / avg_vol;
            } else {
                out[i] = 1.0;  // No spike if average is 0
            }
        }
    }
}

// MFI series kernel (Money Flow Index)
extern "C" __global__ void mfi_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
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
    const double* v_arr = volume + (size_t)s * n;
    double* out = output + (size_t)s * n;

    // NaN for first `period` bars
    for (int i = 0; i <= period; i++) {
        out[i] = 0.0 / 0.0; // NaN
    }

    // Calculate typical prices and flows
    for (int i = period + 1; i < n; i++) {
        double pos_flow = 0.0;
        double neg_flow = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double tp_curr = (h[j] + l[j] + c[j]) / 3.0;
            double tp_prev = (h[j-1] + l[j-1] + c[j-1]) / 3.0;
            double rmf = tp_curr * v_arr[j];
            if (tp_curr > tp_prev) pos_flow += rmf;
            else if (tp_curr < tp_prev) neg_flow += rmf;
        }
        if (neg_flow > 1e-12) {
            double mfr = pos_flow / neg_flow;
            out[i] = 100.0 - 100.0 / (1.0 + mfr);
        } else {
            out[i] = 100.0;
        }
    }
}

// CMF series kernel (Chaikin Money Flow)
extern "C" __global__ void cmf_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
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
    const double* v_arr = volume + (size_t)s * n;
    double* out = output + (size_t)s * n;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = 0.0 / 0.0; // NaN
    }

    for (int i = period - 1; i < n; i++) {
        double sum_mfv = 0.0;
        double sum_vol = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double range = h[j] - l[j];
            double mf_mult = (range > 1e-12) ? ((c[j] - l[j]) - (h[j] - c[j])) / range : 0.0;
            sum_mfv += mf_mult * v_arr[j];
            sum_vol += v_arr[j];
        }
        out[i] = (sum_vol > 1e-12) ? sum_mfv / sum_vol : 0.0;
    }
}

// Fibonacci Pivot series kernel
extern "C" __global__ void fibo_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    double* pivot,
    double* r1,
    double* s1,
    int n,
    int period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;

    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    double* p_out = pivot + (size_t)s * n;
    double* r1_out = r1 + (size_t)s * n;
    double* s1_out = s1 + (size_t)s * n;

    for (int i = 0; i < period - 1 && i < n; i++) {
        p_out[i] = 0.0 / 0.0;
        r1_out[i] = 0.0 / 0.0;
        s1_out[i] = 0.0 / 0.0;
    }

    for (int i = period - 1; i < n; i++) {
        double window_high = -1e30;
        double window_low = 1e30;
        for (int j = i - period + 1; j <= i; j++) {
            if (h[j] > window_high) window_high = h[j];
            if (l[j] < window_low) window_low = l[j];
        }
        double p = (window_high + window_low + c[i]) / 3.0;
        double range = window_high - window_low;
        p_out[i] = p;
        r1_out[i] = p + 0.382 * range;
        s1_out[i] = p - 0.382 * range;
    }
}

// SuperTrend series kernel
extern "C" __global__ void supertrend_series_kernel(
    const double* high,
    const double* low,
    const double* close,
    const double* atr,
    double* st_value,
    double* st_dir,
    int n,
    double multiplier,
    int atr_period,
    int batch
) {
    int s = blockIdx.x;
    if (s >= batch) return;

    const double* h = high + (size_t)s * n;
    const double* l = low + (size_t)s * n;
    const double* c = close + (size_t)s * n;
    const double* a = atr + (size_t)s * n;
    double* sv = st_value + (size_t)s * n;
    double* sd = st_dir + (size_t)s * n;

    for (int i = 0; i < atr_period && i < n; i++) {
        sv[i] = 0.0 / 0.0;
        sd[i] = 0.0;
    }

    if (atr_period >= n) return;

    double final_upper, final_lower;
    double hl2 = (h[atr_period] + l[atr_period]) / 2.0;
    final_upper = hl2 + multiplier * a[atr_period];
    final_lower = hl2 - multiplier * a[atr_period];
    sd[atr_period] = 1.0;
    sv[atr_period] = final_lower;

    for (int i = atr_period + 1; i < n; i++) {
        hl2 = (h[i] + l[i]) / 2.0;
        double basic_upper = hl2 + multiplier * a[i];
        double basic_lower = hl2 - multiplier * a[i];

        final_upper = (basic_upper < final_upper || c[i-1] > final_upper) ? basic_upper : final_upper;
        final_lower = (basic_lower > final_lower || c[i-1] < final_lower) ? basic_lower : final_lower;

        if (sd[i-1] == 1.0) {
            sd[i] = (c[i] < final_lower) ? -1.0 : 1.0;
        } else {
            sd[i] = (c[i] > final_upper) ? 1.0 : -1.0;
        }

        sv[i] = (sd[i] == 1.0) ? final_lower : final_upper;
    }
}

// Kernel to calculate trend indicators
extern "C" __global__ void calculate_trend_kernel(
    const double* close,
    double* trend,
    double* trend_short,
    int n,
    int batch,
    int long_period,
    int short_period
) {
    int s = blockIdx.x;
    if (s >= batch) return;
    
    const double* price = close + (size_t)s * n;
    double* trend_long = trend + (size_t)s * n;
    double* trend_short_out = trend_short + (size_t)s * n;
    
    for (int i = 0; i < n; i++) {
        trend_long[i] = 0.0;
        trend_short_out[i] = 0.0;
        
        if (i >= long_period) {
            // Calculate long-term trend (based on SMA comparison)
            double sum_long = 0.0;
            for (int j = 0; j < long_period; j++) {
                sum_long += price[i - j];
            }
            double sma_long = sum_long / long_period;
            
            trend_long[i] = (price[i] > sma_long) ? 1.0 : -1.0;
        }
        
        if (i >= short_period) {
            // Calculate short-term trend
            double sum_short = 0.0;
            for (int j = 0; j < short_period; j++) {
                sum_short += price[i - j];
            }
            double sma_short = sum_short / short_period;
            
            trend_short_out[i] = (price[i] > sma_short) ? 1.0 : -1.0;
        }
    }
}