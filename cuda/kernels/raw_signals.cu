#include <cuda_runtime.h>
#include <math.h>

#define NUM_SIGNALS 10

// Helper: Sigmoid normalization (matches your Rust logic)
__device__ float sigmoid_normalize(float value, float center, float steepness) {
    float x = (value - center) * steepness;
    return 1.0f / (1.0f + expf(-x));
}

// Helper: Normalize RSI
__device__ float normalize_rsi_score(float rsi) {
    if (isnan(rsi)) return 0.0f;
    float dist_0 = fabsf(rsi - 0.0f);
    float dist_100 = fabsf(rsi - 100.0f);
    float dist = fminf(dist_0, dist_100);
    dist = fminf(dist, 50.0f);
    float norm = 1.0f - (dist / 50.0f);
    return sigmoid_normalize(norm, 0.5f, 8.0f);
}

// -------------------------------------------------------------------------
// COMPREHENSIVE RAW SIGNALS KERNEL
// -------------------------------------------------------------------------
// ... (mapping updated)
// Signal 9: SMA Raw
extern "C" __global__ void calculate_raw_signals_kernel(
    const double** indicators, // Array of pointers to rsi, bb_upper, bb_mid, bb_lower, close, stoch_k, stoch_d, atr, cci, macd_line, macd_histogram, obv, williams_r, sma
    float* output_scores,
    int8_t* output_sides,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Access via index:
    // 0: rsi, 1: bb_upper, 2: bb_mid, 3: bb_lower, 4: close
    // 5: stoch_k, 6: stoch_d, 7: atr, 8: cci, 9: macd_line
    // 10: macd_histogram, 11: obv, 12: williams_r, 13: sma
    const double* rsi = indicators[0];
    const double* bb_upper = indicators[1];
    const double* bb_mid = indicators[2];
    const double* bb_lower = indicators[3];
    const double* close = indicators[4];
    const double* stoch_k = indicators[5];
    const double* stoch_d = indicators[6];
    const double* atr = indicators[7];
    const double* cci = indicators[8];
    const double* macd_line = indicators[9];
    const double* macd_histogram = indicators[10];
    const double* obv = indicators[11];
    const double* williams_r = indicators[12];
    const double* sma = indicators[13];

    // --- Signal 0: RSI Raw ---
    float rsi_val = (float)rsi[idx];
    float rsi_score = normalize_rsi_score(rsi_val);
    int8_t rsi_side = 0;
    if (rsi_val > 70.0f) rsi_side = -1;
    else if (rsi_val < 30.0f) rsi_side = 1;
    output_scores[idx * NUM_SIGNALS + 0] = rsi_score;
    output_sides[idx * NUM_SIGNALS + 0] = rsi_side;

    // --- Signal 1: Bollinger Bands Raw ---
    double p = close[idx];
    double u = bb_upper[idx];
    double m = bb_mid[idx];
    double l = bb_lower[idx];
    float bb_score = 0.0f;
    int8_t bb_side = 0;
    if (!isnan(p) && !isnan(u) && !isnan(l)) {
        double width = u - l;
        if (width > 0.0) {
            if (p >= u * 0.99) { // Touching upper
                float strength = fminf((float)((p - m) / width), 1.0f);
                bb_score = sigmoid_normalize(strength, 0.3f, 6.0f);
                bb_side = -1;
            } else if (p <= l * 1.01) { // Touching lower
                float strength = fminf((float)((m - p) / width), 1.0f);
                bb_score = sigmoid_normalize(strength, 0.3f, 6.0f);
                bb_side = 1;
            }
        }
    }
    output_scores[idx * NUM_SIGNALS + 1] = bb_score;
    output_sides[idx * NUM_SIGNALS + 1] = bb_side;

    // --- Signal 2: Stochastic Raw ---
    float k_val = (float)stoch_k[idx];
    float d_val = (float)stoch_d[idx];
    float stoch_score = 0.0f;
    int8_t stoch_side = 0;
    if (!isnan(k_val) && !isnan(d_val)) {
        if (k_val >= 80.0f && d_val >= 80.0f) {
            float strength = fminf((k_val - 50.0f) / 50.0f, 1.0f);
            stoch_score = sigmoid_normalize(strength, 0.3f, 6.0f);
            stoch_side = -1; // Overbought
        } else if (k_val <= 20.0f && d_val <= 20.0f) {
            float strength = fminf((50.0f - k_val) / 50.0f, 1.0f);
            stoch_score = sigmoid_normalize(strength, 0.3f, 6.0f);
            stoch_side = 1; // Oversold
        }
    }
    output_scores[idx * NUM_SIGNALS + 2] = stoch_score;
    output_sides[idx * NUM_SIGNALS + 2] = stoch_side;

    // --- Signal 3: ATR Raw ---
    float atr_val = (float)atr[idx];
    float atr_score = 0.0f;
    if (!isnan(atr_val) && atr_val > 0.0) {
        float normalized = fminf(atr_val / 0.1, 1.0); // Assuming max meaningful ATR is 10% of price - this is a simplification
        atr_score = sigmoid_normalize(normalized, 0.2f, 5.0f);
    }
    output_scores[idx * NUM_SIGNALS + 3] = atr_score;
    output_sides[idx * NUM_SIGNALS + 3] = 0; // Neutral side

    // --- Signal 4: CCI Raw ---
    float cci_val = (float)cci[idx];
    float cci_score = 0.0f;
    int8_t cci_side = 0;
    if (!isnan(cci_val)) {
        float abs_value = fabsf(cci_val);
        float normalized = fminf(abs_value / 200.0f, 1.0f);
        cci_score = sigmoid_normalize(normalized, 0.3f, 6.0f);
        if (cci_val > 100.0f) {
            cci_side = -1;
        } else if (cci_val < -100.0f) {
            cci_side = 1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 4] = cci_score;
    output_sides[idx * NUM_SIGNALS + 4] = cci_side;

    // --- Signal 5: MACD Line Raw ---
    float macd_line_val = (float)macd_line[idx];
    float macd_line_score = 0.0f;
    int8_t macd_line_side = 0;
    if(!isnan(macd_line_val)) {
        float abs_value = fabsf(macd_line_val);
        float normalized = fminf(abs_value / 10.0f, 1.0f);
        macd_line_score = sigmoid_normalize(normalized, 0.3f, 6.0f);
        if (macd_line_val > 0.0f) {
            macd_line_side = 1;
        } else {
            macd_line_side = -1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 5] = macd_line_score;
    output_sides[idx * NUM_SIGNALS + 5] = macd_line_side;

    // --- Signal 6: MACD Histogram Raw ---
    float macd_hist_val = (float)macd_histogram[idx];
    float macd_hist_score = 0.0f;
    int8_t macd_hist_side = 0;
    if(!isnan(macd_hist_val)) {
        float abs_value = fabsf(macd_hist_val);
        float normalized = fminf(abs_value / 2.0f, 1.0f);
        macd_hist_score = sigmoid_normalize(normalized, 0.2f, 5.0f);
        if (macd_hist_val > 0.0f) {
            macd_hist_side = 1;
        } else {
            macd_hist_side = -1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 6] = macd_hist_score;
    output_sides[idx * NUM_SIGNALS + 6] = macd_hist_side;

    // --- Signal 7: OBV Raw ---
    float obv_val = (float)obv[idx];
    float obv_score = 0.0f;
    int8_t obv_side = 0;
    if (!isnan(obv_val)) {
        float abs_obv = fabsf(obv_val);
        if (abs_obv > 0.0f) {
            float log_obv = logf(abs_obv);
            float normalized = fminf(fmaxf(log_obv / 20.0f, 0.0f), 1.0f);
            obv_score = sigmoid_normalize(normalized, 0.3f, 6.0f);
        }
        if (obv_val > 0.0f) {
            obv_side = 1;
        } else {
            obv_side = -1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 7] = obv_score;
    output_sides[idx * NUM_SIGNALS + 7] = obv_side;

    // --- Signal 8: Williams %R Raw ---
    float williams_val = (float)williams_r[idx];
    float williams_score = 0.0f;
    int8_t williams_side = 0;
    if (!isnan(williams_val)) {
        float abs_value = fabsf(williams_val);
        float distance_from_extremes = fminf(fabsf(abs_value - 0.0f), fabsf(abs_value - 100.0f));
        distance_from_extremes = fminf(distance_from_extremes, 50.0f);
        float normalized = 1.0f - (distance_from_extremes / 50.0f);
        williams_score = sigmoid_normalize(normalized, 0.5f, 8.0f);

        if (williams_val > -20.0f) {
            williams_side = -1;
        } else if (williams_val < -80.0f) {
            williams_side = 1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 8] = williams_score;
    output_sides[idx * NUM_SIGNALS + 8] = williams_side;

    // --- Signal 9: SMA Raw ---
    float price = (float)close[idx];
    float sma_val = (float)sma[idx];
    float sma_score = 0.0f;
    int8_t sma_side = 0;
    if (!isnan(price) && !isnan(sma_val) && sma_val != 0.0f) {
        float distance_pct = ((price - sma_val) / sma_val) * 100.0f;
        float abs_distance = fabsf(distance_pct);
        float normalized = fminf(abs_distance / 10.0f, 1.0f);
        sma_score = sigmoid_normalize(normalized, 0.3f, 6.0f);
        if (distance_pct > 0.0f) {
            sma_side = 1;
        } else {
            sma_side = -1;
        }
    }
    output_scores[idx * NUM_SIGNALS + 9] = sma_score;
    output_sides[idx * NUM_SIGNALS + 9] = sma_side;
}