use tokio::process::Command;

pub struct DockerHealthChecker;

impl DockerHealthChecker {
    pub async fn check_docker_running(&self) -> bool {
        match Command::new("docker")
            .arg("ps")
            .output()
            .await {
                Ok(output) => output.status.success(),
                Err(_) => false,
            }
    }

    pub async fn check_containers_status(&self) -> bool {
        // Non-blocking check
        match Command::new("docker")
            .arg("ps")
            .arg("--format")
            .arg("{{.Names}}")
            .output()
            .await {
                Ok(output) => {
                    if output.stdout.is_empty() {
                        true // No containers running is OK
                    } else {
                        true // Containers are running
                    }
                },
                Err(_) => false,
            }
    }
}