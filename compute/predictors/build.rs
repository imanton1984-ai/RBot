fn main() {
    // If you built xgboost via scripts/build_xgboost_cuda.sh:
    // export XGBOOST_LIB_DIR=.../third_party/xgboost/install/lib
    if let Ok(dir) = std::env::var("XGBOOST_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    }

    // dynamic lib name on linux: libxgboost.so
    println!("cargo:rustc-link-lib=dylib=xgboost");

    // Rebuild if these change
    println!("cargo:rerun-if-env-changed=XGBOOST_LIB_DIR");
}