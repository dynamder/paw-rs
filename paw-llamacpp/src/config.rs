use paw_core::PawConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda(u32),
    Metal,
    Vulkan,
}

#[derive(Debug, Clone)]
pub struct PawLlamaCppConfig {
    pub core: PawConfig,
    pub device: DevicePreference,
    pub n_gpu_layers: i32,
    pub n_ctx: Option<u32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub seed: u32,
    pub max_model_copies: usize,
}

impl Default for PawLlamaCppConfig {
    fn default() -> Self {
        Self {
            core: PawConfig::default(),
            device: DevicePreference::Auto,
            n_gpu_layers: 0,
            n_ctx: None,
            n_threads: None,
            n_threads_batch: None,
            seed: 1234,
            max_model_copies: 1,
        }
    }
}

impl PawLlamaCppConfig {
    pub fn builder() -> PawLlamaCppConfigBuilder {
        PawLlamaCppConfigBuilder::new()
    }
}

#[derive(Debug, Default)]
pub struct PawLlamaCppConfigBuilder {
    core: Option<PawConfig>,
    device: Option<DevicePreference>,
    n_gpu_layers: Option<i32>,
    n_ctx: Option<u32>,
    n_threads: Option<i32>,
    n_threads_batch: Option<i32>,
    seed: Option<u32>,
    max_model_copies: Option<usize>,
}

impl PawLlamaCppConfigBuilder {
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
    pub fn n_gpu_layers(mut self, n: i32) -> Self {
        self.n_gpu_layers = Some(n);
        self
    }
    pub fn n_ctx(mut self, n: u32) -> Self {
        self.n_ctx = Some(n);
        self
    }
    pub fn n_threads(mut self, n: i32) -> Self {
        self.n_threads = Some(n);
        self
    }
    pub fn n_threads_batch(mut self, n: i32) -> Self {
        self.n_threads_batch = Some(n);
        self
    }
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn max_model_copies(mut self, n: usize) -> Self {
        self.max_model_copies = Some(n);
        self
    }

    pub fn build(self) -> PawLlamaCppConfig {
        let defaults = PawLlamaCppConfig::default();
        PawLlamaCppConfig {
            core: self.core.unwrap_or(defaults.core),
            device: self.device.unwrap_or(defaults.device),
            n_gpu_layers: self.n_gpu_layers.unwrap_or(defaults.n_gpu_layers),
            n_ctx: self.n_ctx.or(defaults.n_ctx),
            n_threads: self.n_threads.or(defaults.n_threads),
            n_threads_batch: self.n_threads_batch.or(defaults.n_threads_batch),
            seed: self.seed.unwrap_or(defaults.seed),
            max_model_copies: self.max_model_copies.unwrap_or(defaults.max_model_copies),
        }
    }
}
