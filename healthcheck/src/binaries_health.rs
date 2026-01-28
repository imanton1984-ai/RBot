use std::path::Path;

pub struct BinariesHealthChecker;

impl BinariesHealthChecker {
    pub fn check_binary_exists(&self, binary_name: &str) -> bool {
        Path::new(&format!("./target/release/{}", binary_name)).exists()
    }

    pub fn check_all_binaries(&self) -> bool {
        self.check_binary_exists("connections")
    }
}