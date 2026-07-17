use std::path::PathBuf;

use crate::error::Result;

const DEFAULT_API_URL: &str = "https://programasweights.com";
const CONFIG_FILE_NAME: &str = "config.json";
const CACHE_DIR_NAME: &str = "programasweights";
const CONFIG_DIR_NAME: &str = "programasweights";

/// Configuration for the PAW client and cache.
///
/// Create via [`PawConfig::builder()`] or [`PawConfig::from_env()`].
///
/// # Example
///
/// ```rust
/// use paw_core::PawConfig;
///
/// let config = PawConfig::builder()
///     .api_url("https://custom.instance.com")
///     .api_key("sk-xxx")
///     .n_ctx(4096)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PawConfig {
    api_url: String,
    api_key: Option<String>,
    cache_dir: PathBuf,
    n_ctx: u32,
    n_gpu_layers: i32,
    verbose: bool,
    offline: bool,
}

impl Default for PawConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl PawConfig {
    /// Create a builder for configuring PAW settings.
    pub fn builder() -> PawConfigBuilder {
        PawConfigBuilder::new()
    }

    /// Build config from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        let api_url = std::env::var("PAW_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string());

        let api_key = std::env::var("PAW_API_KEY").ok();
        let cache_dir = std::env::var("PAW_CACHE_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_dir);

        let n_ctx = std::env::var("PAW_N_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);

        let n_gpu_layers = std::env::var("PAW_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(-1);

        let verbose = std::env::var("PAW_VERBOSE")
            .ok()
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);

        let offline = std::env::var("PAW_OFFLINE")
            .ok()
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);

        Self {
            api_url,
            api_key,
            cache_dir,
            n_ctx,
            n_gpu_layers,
            verbose,
            offline,
        }
    }

    // ── Getters ─────────────────────────────────────────────────────────

    pub fn api_url(&self) -> &str {
        &self.api_url
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn n_gpu_layers(&self) -> i32 {
        self.n_gpu_layers
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    pub fn offline(&self) -> bool {
        self.offline
    }

    // ── Derived paths ───────────────────────────────────────────────────

    /// Return the base models cache directory.
    pub fn base_models_dir(&self) -> PathBuf {
        self.cache_dir.join("base_models")
    }

    /// Return the programs cache directory.
    pub fn programs_dir(&self) -> PathBuf {
        self.cache_dir.join("programs")
    }

    /// Return the runtimes cache directory.
    pub fn runtimes_dir(&self) -> PathBuf {
        self.cache_dir.join("runtimes")
    }

    /// Path to the slug→program_id cache file.
    pub fn slug_cache_path(&self) -> PathBuf {
        self.cache_dir.join("slug_cache.json")
    }

    // ── API Key persistence ─────────────────────────────────────────────

    pub fn effective_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| self.load_persisted_api_key())
    }

    fn load_persisted_api_key(&self) -> Option<String> {
        let path = self.config_file_path();
        if !path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
        parsed
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    fn config_file_path(&self) -> PathBuf {
        config_dir().join(CONFIG_FILE_NAME)
    }

    /// Persist the API key to the config file.
    pub fn save_api_key(&self, key: &str) -> std::io::Result<()> {
        let path = self.config_file_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::json!({ "api_key": key });
        std::fs::write(path, serde_json::to_string_pretty(&content)?)?;
        Ok(())
    }
}

/// Builder for [`PawConfig`].
///
/// Fields set on the builder override environment variables;
/// any unset fields fall back to the env-var defaults.
#[derive(Debug, Default)]
pub struct PawConfigBuilder {
    api_url: Option<String>,
    api_key: Option<String>,
    cache_dir: Option<PathBuf>,
    n_ctx: Option<u32>,
    n_gpu_layers: Option<i32>,
    verbose: Option<bool>,
    offline: Option<bool>,
}

impl PawConfigBuilder {
    fn new() -> Self {
        Self::default()
    }

    /// Set the PAW API base URL.
    pub fn api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = Some(url.into());
        self
    }

    /// Set the API key for authenticated requests.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the root cache directory.
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Set the context window size (default 2048).
    pub fn n_ctx(mut self, n: u32) -> Self {
        self.n_ctx = Some(n);
        self
    }

    /// Set GPU layers (-1 = all, 0 = CPU).
    pub fn n_gpu_layers(mut self, n: i32) -> Self {
        self.n_gpu_layers = Some(n);
        self
    }

    /// Enable verbose logging.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = Some(v);
        self
    }

    /// Skip server checks; use local cache only.
    pub fn offline(mut self, v: bool) -> Self {
        self.offline = Some(v);
        self
    }

    /// Build the config, starting from env defaults and applying overrides.
    pub fn build(self) -> Result<PawConfig> {
        let env = PawConfig::from_env();

        Ok(PawConfig {
            api_url: self.api_url.unwrap_or(env.api_url),
            api_key: self.api_key.or(env.api_key),
            cache_dir: self.cache_dir.unwrap_or(env.cache_dir),
            n_ctx: self.n_ctx.unwrap_or(env.n_ctx),
            n_gpu_layers: self.n_gpu_layers.unwrap_or(env.n_gpu_layers),
            verbose: self.verbose.unwrap_or(env.verbose),
            offline: self.offline.unwrap_or(env.offline),
        })
    }
}

fn config_dir() -> PathBuf {
    std::env::var("PAW_CONFIG_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(CONFIG_DIR_NAME)
        })
}

fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CACHE_DIR_NAME)
}

/// Convenience: get the API URL from environment or default.
pub fn get_api_url() -> String {
    PawConfig::from_env().api_url().to_string()
}

/// Convenience: get the API key from environment or config file.
pub fn get_api_key() -> Option<String> {
    PawConfig::from_env().effective_api_key()
}

/// Convenience: save an API key to the config file.
pub fn set_api_key(key: &str) -> std::io::Result<()> {
    let config = PawConfig::from_env();
    config.save_api_key(key)
}
