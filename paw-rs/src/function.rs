use paw_candle::{DevicePreference, PawCandleConfig, PawFnLoader, PawFunction, PawRuntimeOptions};
use paw_core::{CompileRequest, PawBundle, PawClient, PawConfig, Error};

/// Information needed to download the base model and tokenizer for a given interpreter.
struct ModelInfo {
    gguf_owner: &'static str,
    gguf_name: &'static str,
    gguf_file: &'static str,
    tokenizer_owner: &'static str,
    tokenizer_name: &'static str,
}

fn model_info(interpreter: &str) -> Option<ModelInfo> {
    match interpreter {
        "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => Some(ModelInfo {
            gguf_owner: "programasweights",
            gguf_name: "Qwen3-0.6B-GGUF-Q6_K",
            gguf_file: "qwen3-0.6b-q6_k.gguf",
            tokenizer_owner: "Qwen",
            tokenizer_name: "Qwen3-0.6B",
        }),
        "gpt2" | "gpt2-q8_0" => Some(ModelInfo {
            gguf_owner: "programasweights",
            gguf_name: "GPT2-GGUF-Q8_0",
            gguf_file: "gpt2-q8_0.gguf",
            tokenizer_owner: "openai-community",
            tokenizer_name: "gpt2",
        }),
        _ => None,
    }
}

/// High-level PAW function wrapper.
///
/// Construct via [`PawFnBuilder`]:
/// ```rust,no_run
/// use paw_rs::prelude::*;
///
/// # async fn example() -> Result<(), paw_core::Error> {
/// let mut fn = PawFn::builder().slug("email-triage").load().await?;
/// let output = fn.run("Urgent: server is down!")?;
/// println!("{output}");
/// # Ok(())
/// # }
/// ```
pub struct PawFn {
    inner: PawFunction,
}

impl PawFn {
    pub fn builder() -> PawFnBuilder {
        PawFnBuilder::new()
    }

    /// Run inference with default options.
    pub fn run(&mut self, input: &str) -> Result<String, Error> {
        self.inner.run(input, &PawRuntimeOptions::default())
    }

    /// Run inference with custom options.
    pub fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.inner.run(input, opts)
    }
}

/// Builder for loading or compiling a PAW program.
///
/// # Load (existing program)
/// ```rust,no_run
/// use paw_rs::prelude::*;
/// # async fn ex() -> Result<(), paw_core::Error> {
/// let mut fn = PawFn::builder()
///     .config(PawConfig::from_env())
///     .device(DevicePreference::Cpu)
///     .slug("email-triage")
///     .load()
///     .await?;
/// # Ok(()) }
/// ```
///
/// # Compile (new program)
/// ```rust,no_run
/// use paw_rs::prelude::*;
/// # async fn ex() -> Result<(), paw_core::Error> {
/// let mut fn = PawFn::builder()
///     .spec("Classify sentiment")
///     .compiler("paw-4b-qwen3-0.6b")
///     .ephemeral(true)
///     .compile()
///     .await?;
/// # Ok(()) }
/// ```
pub struct PawFnBuilder {
    config: PawConfig,
    device: DevicePreference,
    slug: Option<String>,
    spec: Option<String>,
    compiler: Option<String>,
    ephemeral: bool,
}

impl Default for PawFnBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PawFnBuilder {
    pub fn new() -> Self {
        Self {
            config: PawConfig::from_env(),
            device: DevicePreference::Auto,
            slug: None,
            spec: None,
            compiler: None,
            ephemeral: false,
        }
    }

    pub fn config(mut self, config: PawConfig) -> Self {
        self.config = config;
        self
    }

    pub fn device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    pub fn slug(mut self, slug: impl Into<String>) -> Self {
        self.slug = Some(slug.into());
        self
    }

    pub fn spec(mut self, spec: impl Into<String>) -> Self {
        self.spec = Some(spec.into());
        self
    }

    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }

    pub fn ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    /// Load an existing program by slug.
    pub async fn load(mut self) -> Result<PawFn, Error> {
        let slug = self.slug.take().ok_or_else(|| {
            Error::Config("slug is required for load()".to_string())
        })?;

        let client = PawClient::new(&self.config);
        let program_id = client.resolve_slug(&slug).await?;
        let dir = client.download_paw(&program_id).await?;

        // Auto-download base model and tokenizer before loading.
        let bundle = PawBundle::load_from_dir(&dir)?;
        self.ensure_model_assets(&bundle, &dir).await?;

        let candle_config = self.build_candle_config();
        let inner = PawFnLoader::new(dir)
            .config(candle_config)
            .load()?;
        Ok(PawFn { inner })
    }

    /// Compile a new program from a spec.
    pub async fn compile(mut self) -> Result<PawFn, Error> {
        let spec = self.spec.take().ok_or_else(|| {
            Error::Config("spec is required for compile()".to_string())
        })?;

        let request = {
            let mut b = CompileRequest::builder().spec(spec).ephemeral(self.ephemeral);
            if let Some(ref c) = self.compiler {
                b = b.compiler(c);
            }
            b.build()?
        };

        let client = PawClient::new(&self.config);
        let program = client.compile(request).await?;
        let dir = client.download_paw(&program.id).await?;

        let bundle = PawBundle::load_from_dir(&dir)?;
        self.ensure_model_assets(&bundle, &dir).await?;

        let candle_config = self.build_candle_config();
        let inner = PawFnLoader::new(dir)
            .config(candle_config)
            .load()?;
        Ok(PawFn { inner })
    }

    // ── internal helpers ──────────────────────────────────────────

    fn build_candle_config(&self) -> PawCandleConfig {
        PawCandleConfig::builder()
            .core(self.config.clone())
            .device(self.device)
            .build()
    }

    /// Download the base GGUF model and tokenizer if not already cached.
    async fn ensure_model_assets(&self, bundle: &PawBundle, program_dir: &std::path::Path) -> Result<(), Error> {
        let interpreter = bundle.interpreter_model();
        let info = model_info(interpreter).ok_or_else(|| {
            Error::UnsupportedModel(interpreter.to_string())
        })?;

        let hf = hf_hub::HFClient::new()
            .map_err(|e| Error::Other(format!("HF client: {e}")))?;

        // Download GGUF base model to cache.
        let gguf_path = self.config.base_models_dir().join(info.gguf_file);
        if !gguf_path.exists() {
            if let Some(parent) = gguf_path.parent() {
                std::fs::create_dir_all(parent).map_err(Error::Io)?;
            }
            let tmp = hf
                .model(info.gguf_owner, info.gguf_name)
                .download_file()
                .filename(info.gguf_file)
                .send()
                .await
                .map_err(|e| Error::Other(format!("download GGUF: {e}")))?;
            std::fs::rename(&tmp, &gguf_path).map_err(Error::Io)?;
        }

        // Download tokenizer to the program directory.
        let tokenizer_path = program_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            let tmp = hf
                .model(info.tokenizer_owner, info.tokenizer_name)
                .download_file()
                .filename("tokenizer.json")
                .send()
                .await
                .map_err(|e| Error::Other(format!("download tokenizer: {e}")))?;
            std::fs::rename(&tmp, &tokenizer_path).map_err(Error::Io)?;
        }

        Ok(())
    }
}
