use std::process::Command;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=kernels/indicators.cu");
    println!("cargo:rerun-if-changed=kernels/predictors.cu");
    println!("cargo:rerun-if-changed=kernels/raw_signals.cu");
    println!("cargo:rerun-if-changed=kernels/features.cu");
    println!("cargo:rerun-if-changed=kernels/entry_model.cu");  // Entry policy labeling

    // Проверяем наличие nvcc
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found. CUDA kernels will not be recompiled.");
        return;
    }

    let out_dir = env::var("OUT_DIR").unwrap();

    // Компилируем indicators.cu -> indicators.ptx (WITHOUT --use_fast_math for precision)
    let status_indicators = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/indicators.ptx", out_dir),
            "kernels/indicators.cu",
            "--gpu-architecture=compute_75", // Или compute_60+ для совместимости
            // REMOVED --use_fast_math for indicators to improve precision
        ])
        .status()
        .expect("Failed to execute nvcc for indicators.cu");

    if !status_indicators.success() {
        panic!("NVCC compilation failed for indicators.cu");
    }

    // Компилируем predictors.cu -> predictors.ptx
    let status_predictors = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/predictors.ptx", out_dir),
            "kernels/predictors.cu",
            "--gpu-architecture=compute_75",
            "--use_fast_math"
        ])
        .status()
        .expect("Failed to execute nvcc for predictors.cu");

    if !status_predictors.success() {
        panic!("NVCC compilation failed for predictors.cu");
    }

    // Компилируем raw_signals.cu -> raw_signals.ptx
    let status_raw_signals = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/raw_signals.ptx", out_dir),
            "kernels/raw_signals.cu",
            "--gpu-architecture=compute_75",
            "--use_fast_math"
        ])
        .status()
        .expect("Failed to execute nvcc for raw_signals.cu");

    if !status_raw_signals.success() {
        panic!("NVCC compilation failed for raw_signals.cu");
    }

    // Компилируем features.cu -> features.ptx
    let status_features = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/features.ptx", out_dir),
            "kernels/features.cu",
            "--gpu-architecture=compute_75",
            // REMOVED --use_fast_math for features to maintain precision in feature combination
        ])
        .status()
        .expect("Failed to execute nvcc for features.cu");

    if !status_features.success() {
        panic!("NVCC compilation failed for features.cu");
    }

    // Компилируем entry_model.cu -> entry_model.ptx
    let status_entry = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/entry_model.ptx", out_dir),
            "kernels/entry_model.cu",
            "--gpu-architecture=compute_75",
            // No --use_fast_math for precision in PnL calculations
        ])
        .status()
        .expect("Failed to execute nvcc for entry_model.cu");

    if !status_entry.success() {
        panic!("NVCC compilation failed for entry_model.cu");
    }
}