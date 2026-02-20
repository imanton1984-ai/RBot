// settings/src/exchange_settings.rs
//
// Настройки биржи: API ключи (из ~/.settings.json), комиссии, режим аккаунта.
// API ключи НИКОГДА не хранятся в репозитории — только в скрытом файле home-директории.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, warn, error};

/// Режим аккаунта
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountMode {
    /// Binance USDⓈ-M Futures
    BinanceFutures,
}

impl Default for AccountMode {
    fn default() -> Self {
        Self::BinanceFutures
    }
}

impl std::fmt::Display for AccountMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinanceFutures => write!(f, "binance_futures"),
        }
    }
}

/// Секретные данные API (загружаются из ~/.settings.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCredentials {
    pub api_key: String,
    pub api_secret: String,
}

/// Формат файла ~/.settings.json
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SettingsFile {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_secret: Option<String>,
    // Дополнительные поля из файла (игнорируются)
    #[serde(flatten)]
    _extra: std::collections::HashMap<String, serde_json::Value>,
}

/// Настройки биржи
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeSettings {
    /// Комиссия тейкера (в процентах, напр. 0.04 = 0.04%)
    pub taker_fee: f64,

    /// Комиссия мейкера (в процентах, напр. 0.02 = 0.02%)
    pub maker_fee: f64,

    /// Режим аккаунта
    pub account_mode: AccountMode,

    /// API credentials (не сериализуются в конфиг-файлы проекта)
    #[serde(skip)]
    pub credentials: Option<ApiCredentials>,
}

impl Default for ExchangeSettings {
    fn default() -> Self {
        Self {
            taker_fee: 0.04,
            maker_fee: 0.02,
            account_mode: AccountMode::BinanceFutures,
            credentials: None,
        }
    }
}

impl ExchangeSettings {
    /// Путь к скрытому файлу с API ключами
    fn credentials_path() -> anyhow::Result<PathBuf> {
        let home = std::env::var("HOME")
            .map_err(|_| anyhow::anyhow!("HOME environment variable not set"))?;
        Ok(PathBuf::from(home).join(".settings.json"))
    }

    /// Загрузить API credentials из ~/.settings.json
    pub fn load_credentials() -> anyhow::Result<ApiCredentials> {
        // Сначала проверяем env-переменные (приоритет выше файла)
        if let (Ok(key), Ok(secret)) = (
            std::env::var("BINANCE_API_KEY"),
            std::env::var("BINANCE_API_SECRET"),
        ) {
            info!("API credentials loaded from environment variables");
            return Ok(ApiCredentials {
                api_key: key,
                api_secret: secret,
            });
        }

        // Читаем из ~/.settings.json
        let path = Self::credentials_path()?;
        Self::load_credentials_from_file(&path)
    }

    /// Загрузить credentials из конкретного файла
    pub fn load_credentials_from_file(path: &Path) -> anyhow::Result<ApiCredentials> {
        if !path.exists() {
            anyhow::bail!(
                "Credentials file not found at {:?}. \
                 Create ~/.settings.json with {{\"api_key\": \"...\", \"api_secret\": \"...\"}} \
                 or set BINANCE_API_KEY / BINANCE_API_SECRET env vars.",
                path
            );
        }

        let content = std::fs::read_to_string(path)?;
        let settings_file: SettingsFile = serde_json::from_str(&content)
            .map_err(|e| {
                error!("Failed to parse {:?}: {}", path, e);
                anyhow::anyhow!("Invalid JSON in credentials file: {}", e)
            })?;

        let api_key = settings_file.api_key.ok_or_else(|| {
            anyhow::anyhow!("Missing 'api_key' in {:?}", path)
        })?;
        let api_secret = settings_file.api_secret.ok_or_else(|| {
            anyhow::anyhow!("Missing 'api_secret' in {:?}", path)
        })?;

        if api_key.is_empty() || api_secret.is_empty() {
            anyhow::bail!("API key/secret cannot be empty in {:?}", path);
        }

        info!("API credentials loaded from {:?}", path);
        Ok(ApiCredentials {
            api_key,
            api_secret,
        })
    }

    /// Загрузить настройки биржи из JSON-файла
    pub fn load_from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!("Exchange settings file not found at {:?}, using defaults", path);
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path)?;
        let mut settings: Self = serde_json::from_str(&content)?;

        // Загружаем credentials отдельно (из ~/.settings.json)
        match Self::load_credentials() {
            Ok(creds) => settings.credentials = Some(creds),
            Err(e) => warn!("Could not load API credentials: {}. Trading will be disabled.", e),
        }

        info!(
            "Loaded exchange settings: taker_fee={}%, maker_fee={}%, mode={}",
            settings.taker_fee, settings.maker_fee, settings.account_mode,
        );
        settings.validate()?;
        Ok(settings)
    }

    /// Загрузить из стандартного пути config/exchange_settings.json
    pub fn load() -> anyhow::Result<Self> {
        Self::load_from_file("config/exchange_settings.json")
    }

    /// Загрузить с override из env-переменных
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut settings = Self::load()?;

        if let Ok(v) = std::env::var("EXCHANGE_TAKER_FEE") {
            if let Ok(n) = v.parse() {
                settings.taker_fee = n;
            }
        }
        if let Ok(v) = std::env::var("EXCHANGE_MAKER_FEE") {
            if let Ok(n) = v.parse() {
                settings.maker_fee = n;
            }
        }

        settings.validate()?;
        Ok(settings)
    }

    /// Валидация настроек
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.taker_fee < 0.0 || self.taker_fee > 1.0 {
            anyhow::bail!(
                "Invalid taker_fee: {}%. Must be 0.0-1.0%",
                self.taker_fee
            );
        }
        if self.maker_fee < 0.0 || self.maker_fee > 1.0 {
            anyhow::bail!(
                "Invalid maker_fee: {}%. Must be 0.0-1.0%",
                self.maker_fee
            );
        }
        Ok(())
    }

    /// Проверить наличие API credentials
    pub fn has_credentials(&self) -> bool {
        self.credentials.is_some()
    }

    /// Получить API key (паникует если нет credentials)
    pub fn api_key(&self) -> &str {
        self.credentials
            .as_ref()
            .expect("API credentials not loaded. Call load_credentials() first.")
            .api_key
            .as_str()
    }

    /// Получить API secret (паникует если нет credentials)
    pub fn api_secret(&self) -> &str {
        self.credentials
            .as_ref()
            .expect("API credentials not loaded. Call load_credentials() first.")
            .api_secret
            .as_str()
    }

    /// Рассчитать комиссию тейкера для суммы
    pub fn calc_taker_fee(&self, notional_usdt: f64) -> f64 {
        notional_usdt * self.taker_fee / 100.0
    }

    /// Рассчитать комиссию мейкера для суммы
    pub fn calc_maker_fee(&self, notional_usdt: f64) -> f64 {
        notional_usdt * self.maker_fee / 100.0
    }

    /// Сохранить настройки (без credentials!) в JSON-файл
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        // Создаём копию без credentials для сохранения
        let safe_settings = ExchangeSettings {
            taker_fee: self.taker_fee,
            maker_fee: self.maker_fee,
            account_mode: self.account_mode,
            credentials: None, // Никогда не сохраняем ключи в конфиг проекта
        };
        let content = serde_json::to_string_pretty(&safe_settings)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_exchange_settings() {
        let settings = ExchangeSettings::default();
        assert!((settings.taker_fee - 0.04).abs() < 1e-8);
        assert!((settings.maker_fee - 0.02).abs() < 1e-8);
        assert_eq!(settings.account_mode, AccountMode::BinanceFutures);
        assert!(!settings.has_credentials());
    }

    #[test]
    fn test_validate_fees() {
        let mut settings = ExchangeSettings::default();
        settings.taker_fee = -0.01;
        assert!(settings.validate().is_err());

        settings.taker_fee = 1.5;
        assert!(settings.validate().is_err());

        settings.taker_fee = 0.04;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_calc_fees() {
        let settings = ExchangeSettings::default();
        // 0.04% of 1000 USDT = 0.4 USDT
        let fee = settings.calc_taker_fee(1000.0);
        assert!((fee - 0.4).abs() < 1e-8);

        // 0.02% of 1000 USDT = 0.2 USDT
        let fee = settings.calc_maker_fee(1000.0);
        assert!((fee - 0.2).abs() < 1e-8);
    }

    #[test]
    fn test_serde_roundtrip() {
        let settings = ExchangeSettings::default();
        let json = serde_json::to_string_pretty(&settings).unwrap();
        let deserialized: ExchangeSettings = serde_json::from_str(&json).unwrap();
        assert!((deserialized.taker_fee - settings.taker_fee).abs() < 1e-8);
        assert_eq!(deserialized.account_mode, settings.account_mode);
    }

    #[test]
    fn test_credentials_not_serialized() {
        let mut settings = ExchangeSettings::default();
        settings.credentials = Some(ApiCredentials {
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
        });
        let json = serde_json::to_string(&settings).unwrap();
        // credentials should NOT appear in JSON
        assert!(!json.contains("test_key"));
        assert!(!json.contains("test_secret"));
    }
}
