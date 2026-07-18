use paw_core::PawConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

#[derive(Debug, Clone)]
pub struct PawMistralRsConfig {
    pub core: PawConfig,
    pub device: DevicePreference,
    pub base_model_repo: Option<String>,
    pub gguf_filename: Option<String>,
    pub use_isq: bool,
}

impl Default for PawMistralRsConfig {
    fn default() -> Self {
        Self {
            core: PawConfig::default(),
            device: DevicePreference::Auto,
            base_model_repo: None,
            gguf_filename: None,
            use_isq: false,
        }
    }
}

impl PawMistralRsConfig {
    pub fn builder() -> PawMistralRsConfigBuilder {
        PawMistralRsConfigBuilder::new()
    }

    pub fn qwen3_06b() -> Self {
        Self {
            base_model_repo: Some("programasweights/Qwen3-0.6B-GGUF-Q6_K".into()),
            gguf_filename: Some("qwen3-0.6b-q6_k.gguf".into()),
            ..Default::default()
        }
    }

    pub fn gpt2() -> Self {
        Self {
            base_model_repo: Some("programasweights/GPT2-GGUF-Q8_0".into()),
            gguf_filename: Some("gpt2-q8_0.gguf".into()),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct PawMistralRsConfigBuilder {
    core: Option<PawConfig>,
    device: Option<DevicePreference>,
    base_model_repo: Option<String>,
    gguf_filename: Option<String>,
    use_isq: Option<bool>,
}

impl PawMistralRsConfigBuilder {
    fn new() -> Self {
        Self::default()
    }

    pub fn core(mut self, core: PawConfig) -> Self {
        self.core = Some(core);
        self
    }
    pub fn device(mut self, device: DevicePreference) -> Self {
        self.device = Some(device);
        self
    }
    pub fn base_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.base_model_repo = Some(repo.into());
        self
    }
    pub fn gguf_filename(mut self, filename: impl Into<String>) -> Self {
        self.gguf_filename = Some(filename.into());
        self
    }
    pub fn use_isq(mut self, v: bool) -> Self {
        self.use_isq = Some(v);
        self
    }

    pub fn build(self) -> PawMistralRsConfig {
        let defaults = PawMistralRsConfig::default();
        PawMistralRsConfig {
            core: self.core.unwrap_or(defaults.core),
            device: self.device.unwrap_or(defaults.device),
            base_model_repo: self.base_model_repo.or(defaults.base_model_repo),
            gguf_filename: self.gguf_filename.or(defaults.gguf_filename),
            use_isq: self.use_isq.unwrap_or(defaults.use_isq),
        }
    }
}
