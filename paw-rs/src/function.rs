use std::marker::PhantomData;

use paw_candle::{DevicePreference, PawCandleConfig, PawFnLoader, PawFunction, PawRuntimeOptions};
use paw_core::{CompileRequest, PawBundle, PawClient, PawConfig, Error};

// ── Model info ───────────────────────────────────────────────────────

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

// ── State markers ─────────────────────────────────────────────────────

/// Initial builder state: no slug or spec set yet.
pub struct Unset;

/// Builder has a slug, can call `.load()`.
pub struct ForLoad;

/// Builder has a spec, can call `.compile()`.
pub struct ForCompile;

// ── PawFn ─────────────────────────────────────────────────────────────

/// High-level PAW function wrapper.
pub struct PawFn {
    inner: PawFunction,
}

impl PawFn {
    pub fn from_inner(inner: PawFunction) -> Self {
        Self { inner }
    }

    pub fn builder() -> PawFnBuilder<Unset> {
        PawFnBuilder::new()
    }

    pub fn run(&mut self, input: &str) -> Result<String, Error> {
        self.inner.run(input, &PawRuntimeOptions::default())
    }

    pub fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.inner.run(input, opts)
    }
}

// ── PawFnBuilder ──────────────────────────────────────────────────────

pub struct PawFnBuilder<State = Unset> {
    config: PawConfig,
    device: DevicePreference,
    slug: Option<String>,
    spec: Option<String>,
    compiler: Option<String>,
    ephemeral: bool,
    _marker: PhantomData<State>,
}

// ── Common methods (any state) ──────────────────────────────────────

impl<State> PawFnBuilder<State> {
    pub fn config(mut self, config: PawConfig) -> Self {
        self.config = config;
        self
    }

    pub fn device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }
}

// ── Initial state — transitions ─────────────────────────────────────

impl PawFnBuilder<Unset> {
    pub fn new() -> Self {
        Self {
            config: PawConfig::from_env(),
            device: DevicePreference::Auto,
            slug: None,
            spec: None,
            compiler: None,
            ephemeral: false,
            _marker: PhantomData,
        }
    }

    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }

    pub fn ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    pub fn slug(self, slug: impl Into<String>) -> PawFnBuilder<ForLoad> {
        PawFnBuilder {
            config: self.config,
            device: self.device,
            slug: Some(slug.into()),
            spec: None,
            compiler: self.compiler,
            ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }

    pub fn spec(self, spec: impl Into<String>) -> PawFnBuilder<ForCompile> {
        PawFnBuilder {
            config: self.config,
            device: self.device,
            slug: None,
            spec: Some(spec.into()),
            compiler: self.compiler,
            ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
}

// ── Load mode ──────────────────────────────────────────────────────

impl PawFnBuilder<ForLoad> {
    pub async fn load(self) -> Result<PawFn, Error> {
        let slug = self.slug.expect("slug must be set in ForLoad state");
        let client = PawClient::new(&self.config);
        let program_id = client.resolve_slug(&slug).await?;
        let dir = client.download_paw(&program_id).await?;
        download_assets(&self.config, &dir).await?;
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::builder().core(self.config).device(self.device).build())
            .load()?;
        Ok(PawFn { inner })
    }
}

// ── Compile mode ──────────────────────────────────────────────────

impl PawFnBuilder<ForCompile> {
    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }

    pub fn ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    pub async fn compile(self) -> Result<PawFn, Error> {
        let spec = self.spec.expect("spec must be set in ForCompile state");
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
        download_assets(&self.config, &dir).await?;
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::builder().core(self.config).device(self.device).build())
            .load()?;
        Ok(PawFn { inner })
    }
}

// ── Shared helpers ────────────────────────────────────────────────────

async fn download_assets(config: &PawConfig, program_dir: &std::path::Path) -> Result<(), Error> {
    let bundle = PawBundle::load_from_dir(program_dir)?;
    let interpreter = bundle.interpreter_model();
    let info = model_info(interpreter).ok_or_else(|| Error::UnsupportedModel(interpreter.to_string()))?;

    let hf = hf_hub::HFClient::new().map_err(|e| Error::Other(format!("HF client: {e}")))?;

    let gguf_path = config.base_models_dir().join(info.gguf_file);
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
