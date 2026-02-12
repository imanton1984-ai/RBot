use std::process::Command;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=kernels/indicators.cu");
    println!("cargo:rerun-if-changed=kernels/predictors.cu");
    println!("cargo:rerun-if-changed=kernels/raw_signals.cu");

    // Проверяем наличие nvcc
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found. CUDA kernels will not be recompiled.");
        return;
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Компилируем indicators.cu -> indicators.ptx
    let status_indicators = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/indicators.ptx", out_dir),
            "kernels/indicators.cu",
            "--gpu-architecture=compute_75", // Или compute_60+ для совместимости
            "--use_fast_math"
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
}