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