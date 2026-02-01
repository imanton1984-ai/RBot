// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // Get the CUDA installation path
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let _cuda_include = PathBuf::from(&cuda_path).join("include");
    let cuda_lib = if PathBuf::from(&cuda_path).join("lib64").exists() {
        PathBuf::from(&cuda_path).join("lib64")
    } else {
        PathBuf::from(&cuda_path).join("lib")
    };

    // Tell cargo to look for CUDA libraries in the specified directory
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    // Link to the CUDA runtime and driver libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=build.rs");
}