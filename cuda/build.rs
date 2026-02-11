use std::process::Command;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=kernels/indicators.cu");
    println!("cargo:rerun-if-changed=kernels/predictors.cu");

    // Проверяем наличие nvcc
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found. CUDA kernels will not be recompiled.");
        return;
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Компилируем indicators.cu -> indicators.ptx
    let status = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-o", &format!("{}/indicators.ptx", out_dir),
            "kernels/indicators.cu",
            "--gpu-architecture=compute_75", // Или compute_60+ для совместимости
            "--use_fast_math"
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("NVCC compilation failed for indicators.cu");
    }
}