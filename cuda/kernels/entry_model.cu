// cuda/kernels/entry_model.cu
//
// CUDA Kernel for Accelerated Entry Policy Labeling
//
// This kernel accelerates the expert labeling process by parallelizing
// the search for the optimal entry bar within the entry window.
//
// WORKFLOW:
//   1. Each thread processes one setup (signal)
//   2. For each setup, scan the entry window [t0 .. t0+W]
//   3. For each candidate entry bar, simulate PnL with TP/SL
//   4. Find the bar with maximum PnL
//   5. Output: best_entry_offset, best_pnl
//
// USAGE:
//   - Compile: nvcc -ptx -o entry_model.ptx entry_model.cu
//   - Load PTX in Rust via cudarc
//   - Launch: gridDim = (n_setups + blockDim - 1) / blockDim
//
// This is useful when you have millions of signals to label.

#include <cuda_runtime.h>

/// Small buffer for breakeven trailing SL (0.1% inside profit)
#define BREAKEVEN_BUFFER 0.001

/// Position close fractions for partial take-profit
#define TP1_CLOSE_PCT 0.50
#define TP2_CLOSE_PCT 0.30
#define TP3_CLOSE_PCT 0.20

/// OHLCV bar with ATR
typedef struct {
    double open;
    double high;
    double low;
    double close;
    double atr;
} OhlcBar;

/// Simulation configuration
typedef struct {
    int window_bars;
    int max_hold_bars;
    double sl_atr_mult;
    double rr1;
    double rr2;
    double rr3;
    double tp1_close_pct;
    double tp2_close_pct;
    double tp3_close_pct;
} SimCfg;

/// Simulate PnL for a trade entered at entry_idx with partial close logic
/// Returns PnL in percentage
__device__ double simulate_pnl_pct(
    int side,           // +1 long, -1 short
    int entry_idx,
    const OhlcBar* series,
    int series_len,
    SimCfg cfg
) {
    if (entry_idx >= series_len) {
        return 0.0;
    }

    double entry_price = series[entry_idx].close;
    double atr = series[entry_idx].atr;
    if (atr < 1e-9) atr = 1e-9;

    double sl_dist = cfg.sl_atr_mult * atr;

    // Calculate SL and TP levels
    double sl_price = (side > 0) ? (entry_price - sl_dist) : (entry_price + sl_dist);
    double tp1_price = (side > 0) ? (entry_price + cfg.rr1 * sl_dist) : (entry_price - cfg.rr1 * sl_dist);
    double tp2_price = (side > 0) ? (entry_price + cfg.rr2 * sl_dist) : (entry_price - cfg.rr2 * sl_dist);
    double tp3_price = (side > 0) ? (entry_price + cfg.rr3 * sl_dist) : (entry_price - cfg.rr3 * sl_dist);

    // Calculate exit bar
    int exit_idx = entry_idx + cfg.max_hold_bars;
    if (exit_idx >= series_len) exit_idx = series_len - 1;

    double position = 1.0;
    double pnl = 0.0;
    int tp1_hit = 0;
    int tp2_hit = 0;
    int tp3_hit = 0;

    for (int i = entry_idx; i <= exit_idx; i++) {
        double bar_high = series[i].high;
        double bar_low = series[i].low;

        // Check SL hit first
        int sl_hit = (side > 0) ? (bar_low <= sl_price) : (bar_high >= sl_price);
        if (sl_hit) {
            double price_diff = sl_price - entry_price;
            pnl += position * price_diff / entry_price * ((double)side);
            return pnl * 100.0;
        }

        // Check TP1 hit (partial close 50%)
        int tp1_hit_now = (side > 0) ? (bar_high >= tp1_price) : (bar_low <= tp1_price);
        if (tp1_hit_now && !tp1_hit && position > 0.0) {
            double price_diff = tp1_price - entry_price;
            pnl += cfg.tp1_close_pct * price_diff / entry_price * ((double)side);
            position -= cfg.tp1_close_pct;
            tp1_hit = 1;
        }

        // Check TP2 hit (partial close 30%)
        int tp2_hit_now = (side > 0) ? (bar_high >= tp2_price) : (bar_low <= tp2_price);
        if (tp2_hit_now && !tp2_hit && position > 0.0) {
            double price_diff = tp2_price - entry_price;
            pnl += cfg.tp2_close_pct * price_diff / entry_price * ((double)side);
            position -= cfg.tp2_close_pct;
            tp2_hit = 1;
        }

        // Check TP3 hit (close remaining 20%)
        int tp3_hit_now = (side > 0) ? (bar_high >= tp3_price) : (bar_low <= tp3_price);
        if (tp3_hit_now && !tp3_hit && position > 0.0) {
            double price_diff = tp3_price - entry_price;
            pnl += cfg.tp3_close_pct * price_diff / entry_price * ((double)side);
            position -= cfg.tp3_close_pct;
            tp3_hit = 1;
            return pnl * 100.0;
        }
    }

    // Expired: close remaining at exit bar's close
    double exit_price = series[exit_idx].close;
    double price_diff = exit_price - entry_price;
    pnl += position * price_diff / entry_price * ((double)side);
    return pnl * 100.0;
}

/// Find the best entry bar within the window for a given setup
/// Each thread processes one setup
extern "C" __global__ void find_best_entry_kernel(
    const OhlcBar* series,        // [n_bars] OHLCV series with ATR
    int n_bars,
    const int* setup_t0,          // [n_setups] Start bar index for each setup
    const int8_t* setup_side,     // [n_setups] Side (+1 long, -1 short)
    SimCfg cfg,
    int* out_best_offset,         // [n_setups] Best entry offset (or -1 if CANCEL)
    double* out_best_pnl          // [n_setups] Best PnL achieved
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    // if (s >= n_setups) return; // handled by grid size

    int t0 = setup_t0[s];
    int8_t side = setup_side[s];

    double best_pnl = -1e9;
    int best_offset = -1;

    int last_entry = t0 + cfg.window_bars;
    if (last_entry >= n_bars) last_entry = n_bars - 1;

    // Scan entry window
    for (int offset = 0; offset <= (last_entry - t0); offset++) {
        int entry_idx = t0 + offset;
        double pnl = simulate_pnl_pct(side, entry_idx, series, n_bars, cfg);

        if (pnl > best_pnl) {
            best_pnl = pnl;
            best_offset = offset;
        }
    }

    // If best PnL is not profitable, mark as CANCEL
    if (best_pnl <= 0.0) {
        best_offset = -1;
    }

    out_best_offset[s] = best_offset;
    out_best_pnl[s] = best_pnl;
}

/// Batch kernel for scanning multiple setups with variable window sizes
/// More flexible version that handles different timeframes
extern "C" __global__ void find_best_entry_batch_kernel(
    const OhlcBar* series,        // [n_bars] OHLCV series with ATR
    int n_bars,
    const int* setup_t0,          // [n_setups] Start bar index for each setup
    const int8_t* setup_side,     // [n_setups] Side (+1 long, -1 short)
    const int* setup_window,      // [n_setups] Window size per setup (varies by TF)
    const int* setup_max_hold,    // [n_setups] Max hold bars per setup
    double sl_atr_mult,
    double rr1, double rr2, double rr3,
    int* out_best_offset,         // [n_setups] Best entry offset (or -1 if CANCEL)
    double* out_best_pnl          // [n_setups] Best PnL achieved
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;

    int t0 = setup_t0[s];
    int8_t side = setup_side[s];
    int window_bars = setup_window[s];
    int max_hold_bars = setup_max_hold[s];

    SimCfg cfg;
    cfg.window_bars = window_bars;
    cfg.max_hold_bars = max_hold_bars;
    cfg.sl_atr_mult = sl_atr_mult;
    cfg.rr1 = rr1;
    cfg.rr2 = rr2;
    cfg.rr3 = rr3;
    cfg.tp1_close_pct = TP1_CLOSE_PCT;
    cfg.tp2_close_pct = TP2_CLOSE_PCT;
    cfg.tp3_close_pct = TP3_CLOSE_PCT;

    double best_pnl = -1e9;
    int best_offset = -1;

    int last_entry = t0 + window_bars;
    if (last_entry >= n_bars) last_entry = n_bars - 1;

    for (int offset = 0; offset <= (last_entry - t0); offset++) {
        int entry_idx = t0 + offset;
        double pnl = simulate_pnl_pct(side, entry_idx, series, n_bars, cfg);

        if (pnl > best_pnl) {
            best_pnl = pnl;
            best_offset = offset;
        }
    }

    if (best_pnl <= 0.0) {
        best_offset = -1;
    }

    out_best_offset[s] = best_offset;
    out_best_pnl[s] = best_pnl;
}

/// Utility kernel: fill output array with a constant value
extern "C" __global__ void fill_int_kernel(int* out, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

/// Utility kernel: copy OhlcBar from host to device (batch)
extern "C" __global__ void copy_ohlc_kernel(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    const double* atr,
    OhlcBar* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].open = open[idx];
        out[idx].high = high[idx];
        out[idx].low = low[idx];
        out[idx].close = close[idx];
        out[idx].atr = atr[idx];
    }
}
