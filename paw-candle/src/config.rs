use paw_core::PawConfig;

/// Device type for tensor computation.
///
/// `Cuda` and `Metal` variants only exist when the corresponding
/// Cargo feature (`cuda` / `metal`) is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Auto-detect: CUDA > Metal > CPU
    Auto,
    /// Force CPU-only computation.
    Cpu,
    /// NVIDIA GPU acceleration (`--features cuda`).
    #[cfg(feature = "cuda")]
    Cuda,
    /// Apple Silicon GPU acceleration (`--features metal`).
    #[cfg(feature = "metal")]
    Metal,
}

/// Configuration for the candle-based inference runtime.
///
/// Create via [`PawCandleConfig::builder()`] or [`PawCandleConfig::default()`].
///
/// # Example
///
/// ```rust
/// use paw_candle::PawCandleConfig;
///
/// let config = PawCandleConfig::builder()
///     .device(paw_candle::DevicePreference::Cpu)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PawCandleConfig {
    /// PAW SDK core configuration (cache dir, API URL, etc.).
    pub core: PawConfig,
    /// Device preference.
    pub device: DevicePreference,
    /// Whether to load the base model as a GGUF file (default: `true`).
    pub use_gguf: bool,
    /// Override the HuggingFace model repository (e.g. `"programasweights/Qwen3-0.6B-GGUF-Q6_K"`).
    pub base_model_repo: Option<String>,
    /// Override the GGUF filename (e.g. `"qwen3-0.6b-q6_k.gguf"`).
    pub gguf_filename: Option<String>,
}

impl Default for PawCandleConfig {
    fn default() -> Self {
        Self {
            core: PawConfig::default(),
            device: DevicePreference::Auto,
            use_gguf: true,
            base_model_repo: None,
            gguf_filename: None,
        }
    }
}

impl PawCandleConfig {
    /// Create a builder for [`PawCandleConfig`].
    pub fn builder() -> PawCandleConfigBuilder {
        PawCandleConfigBuilder::new()
    }

    /// Pre-configured config for Qwen3-0.6B.
    pub fn qwen3_06b() -> Self {
        Self {
            use_gguf: true,
            base_model_repo: Some("programasweights/Qwen3-0.6B-GGUF-Q6_K".into()),
            gguf_filename: Some("qwen3-0.6b-q6_k.gguf".into()),
            ..Default::default()
        }
    }

    /// Pre-configured config for GPT-2 (124M).
    pub fn gpt2() -> Self {
        Self {
            use_gguf: true,
            base_model_repo: Some("programasweights/GPT2-GGUF-Q8_0".into()),
            gguf_filename: Some("gpt2-q8_0.gguf".into()),
            ..Default::default()
        }
    }
}

/// Builder for [`PawCandleConfig`].
#[derive(Debug, Default)]
pub struct PawCandleConfigBuilder {
    core: Option<PawConfig>,
    device: Option<DevicePreference>,
    use_gguf: Option<bool>,
    base_model_repo: Option<String>,
    gguf_filename: Option<String>,
}

impl PawCandleConfigBuilder {
    fn new() -> Self {
        Self::default()
    }

    /// Set the PAW SDK core configuration.
    pub fn core(mut self, core: PawConfig) -> Self {
        self.core = Some(core);
        self
    }

    /// Set the compute device.
    pub fn device(mut self, device: DevicePreference) -> Self {
        self.device = Some(device);
        self
    }

    /// Whether to load the base model as a GGUF file.
    pub fn use_gguf(mut self, v: bool) -> Self {
        self.use_gguf = Some(v);
        self
    }

    /// Override the HuggingFace model repository.
    pub fn base_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.base_model_repo = Some(repo.into());
        self
    }

    /// Override the GGUF filename.
    pub fn gguf_filename(mut self, filename: impl Into<String>) -> Self {
        self.gguf_filename = Some(filename.into());
        self
    }

    /// Build the [`PawCandleConfig`].
    pub fn build(self) -> PawCandleConfig {
        let defaults = PawCandleConfig::default();

        PawCandleConfig {
            core: self.core.unwrap_or(defaults.core),
            device: self.device.unwrap_or(defaults.device),
            use_gguf: self.use_gguf.unwrap_or(defaults.use_gguf),
            base_model_repo: self.base_model_repo.or(defaults.base_model_repo),
            gguf_filename: self.gguf_filename.or(defaults.gguf_filename),
        }
    }
}
