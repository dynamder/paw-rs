use std::marker::PhantomData;

use paw_core::{CompileRequest, PawClient, PawConfig, Error, PawFnTrait, PawRuntimeOptions};

#[cfg(all(feature = "llamacpp", not(feature = "candle")))]
fn backend_load(dir: std::path::PathBuf, config: PawConfig) -> Result<Box<dyn PawFnTrait>, Error> {
    use paw_llamacpp::{PawLlamaCppConfig, PawFnLoader};
    let inner = PawFnLoader::new(dir)
        .config(PawLlamaCppConfig::builder().core(config).build())
        .load()?;
    Ok(inner as Box<dyn PawFnTrait>)
}

#[cfg(feature = "candle")]
fn backend_load(dir: std::path::PathBuf, config: PawConfig) -> Result<Box<dyn PawFnTrait>, Error> {
    use paw_candle::{DevicePreference, PawCandleConfig};
    let inner = paw_candle::PawFnLoader::new(dir)
        .config(PawCandleConfig::builder().core(config).device(DevicePreference::Auto).build())
        .load()?;
    Ok(Box::new(inner) as Box<dyn PawFnTrait>)
}

macro_rules! cfg_backend_load {
    ($self:ident, $dir:ident) => {
        backend_load($dir, $self.config)
    };
}

// ── State markers ─────────────────────────────────────────────────────

pub struct Unset;
pub struct ForLoad;
pub struct ForCompile;

// ── PawFn ─────────────────────────────────────────────────────────────

/// Typed PAW function (requires `candle` feature).
///
/// Only available when the `candle` backend is enabled.
#[cfg(feature = "candle")]
pub struct PawFn<T: paw_candle::InterpreterModel> {
    inner: paw_candle::PawFunction,
    _phantom: PhantomData<T>,
}

#[cfg(feature = "candle")]
impl<T: paw_candle::InterpreterModel> PawFn<T> {
    pub fn from_inner(inner: paw_candle::PawFunction) -> Self {
        Self { inner, _phantom: PhantomData }
    }

    pub fn run(&mut self, input: &str) -> Result<String, Error> {
        self.inner.run(input, &PawRuntimeOptions::default())
    }

    pub fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.inner.run(input, opts)
    }

    pub fn interpreter(&self) -> &str {
        self.inner.interpreter()
    }

    pub async fn load_slug(slug: &str) -> Result<Self, Error> {
        use paw_candle::{get_or_load_model, DevicePreference, PawCandleConfig};
        let config = PawConfig::from_env();
        let client = PawClient::new(&config);
        let program_id = client.resolve_slug(slug).await?;
        let dir = client.download_paw(&program_id).await?;
        download_assets(&config, &dir).await?;
        let candle_config = PawCandleConfig::builder()
            .core(config.clone())
            .device(DevicePreference::Auto)
            .build();
        let device = paw_candle::runtime::select_device_for_loading(&candle_config)?;
        let shared = get_or_load_model::<T>(&config, &device)?;
        let inner = paw_candle::PawFnLoader::new(dir)
            .config(candle_config)
            .load_with_model(shared)?;
        Ok(Self { inner, _phantom: PhantomData })
    }

    pub async fn compile_spec(spec: &str, compiler: &str) -> Result<Self, Error> {
        use paw_candle::{get_or_load_model, DevicePreference, PawCandleConfig};
        let config = PawConfig::from_env();
        let client = PawClient::new(&config);
        let request = CompileRequest::builder()
            .spec(spec).compiler(compiler).ephemeral(true).build()?;
        let program = client.compile(request).await?;
        let dir = client.download_paw(&program.id).await?;
        download_assets(&config, &dir).await?;
        let candle_config = PawCandleConfig::builder()
            .core(config.clone())
            .device(DevicePreference::Auto)
            .build();
        let device = paw_candle::runtime::select_device_for_loading(&candle_config)?;
        let shared = get_or_load_model::<T>(&config, &device)?;
        let inner = paw_candle::PawFnLoader::new(dir)
            .config(candle_config)
            .load_with_model(shared)?;
        Ok(Self { inner, _phantom: PhantomData })
    }
}

#[cfg(feature = "candle")]
impl<T: paw_candle::InterpreterModel> PawFnTrait for PawFn<T> {
    fn run(&mut self, input: &str) -> Result<String, Error> { self.run(input) }
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> { self.run_with(input, opts) }
    fn interpreter(&self) -> &str { self.interpreter() }
}

// ── PawFnBuilder ──────────────────────────────────────────────────────

pub struct PawFnBuilder<State = Unset> {
    config: PawConfig,
    slug: Option<String>,
    program_id: Option<String>,
    spec: Option<String>,
    compiler: Option<String>,
    ephemeral: bool,
    _marker: PhantomData<State>,
}

impl<State> PawFnBuilder<State> {
    pub fn config(mut self, config: PawConfig) -> Self { self.config = config; self }
}

impl PawFnBuilder<Unset> {
    pub fn builder() -> Self { Self::default() }
    pub fn new() -> Self { Self::default() }

    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into()); self
    }
    pub fn ephemeral(mut self, e: bool) -> Self { self.ephemeral = e; self }
    pub fn slug(self, slug: impl Into<String>) -> PawFnBuilder<ForLoad> {
        PawFnBuilder {
            config: self.config, slug: Some(slug.into()), program_id: None,
            spec: None, compiler: self.compiler, ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
    pub fn id(self, program_id: impl Into<String>) -> PawFnBuilder<ForLoad> {
        PawFnBuilder {
            config: self.config, program_id: Some(program_id.into()), slug: None,
            spec: None, compiler: self.compiler, ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
    pub fn spec(self, spec: impl Into<String>) -> PawFnBuilder<ForCompile> {
        PawFnBuilder {
            config: self.config, slug: None, program_id: None, spec: Some(spec.into()),
            compiler: self.compiler, ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
}

impl Default for PawFnBuilder<Unset> {
    fn default() -> Self {
        Self {
            config: PawConfig::from_env(), slug: None, program_id: None,
            spec: None, compiler: None, ephemeral: false, _marker: PhantomData,
        }
    }
}

// ── Load mode ──────────────────────────────────────────────────────

#[cfg(any(feature = "candle", feature = "llamacpp"))]
impl PawFnBuilder<ForLoad> {
    pub async fn load(self) -> Result<Box<dyn PawFnTrait>, Error> {
        let client = PawClient::new(&self.config);
        let program_id = match (self.slug.as_deref(), self.program_id.as_deref()) {
            (Some(slug), _) => client.resolve_slug(slug).await?,
            (_, Some(id)) => id.to_string(),
            (None, None) => return Err(Error::Other("requires .slug() or .id() before .load()".into())),
        };
        let dir = client.download_paw(&program_id).await?;
        download_assets(&self.config, &dir).await?;

        cfg_backend_load!(self, dir)
    }
}

// ── Compile mode ──────────────────────────────────────────────────

impl PawFnBuilder<ForCompile> {
    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into()); self
    }
    pub fn ephemeral(mut self, e: bool) -> Self { self.ephemeral = e; self }

    pub async fn compile(self) -> Result<Box<dyn PawFnTrait>, Error> {
        let spec = self.spec.expect("spec must be set for ForCompile");
        let request = {
            let mut b = CompileRequest::builder().spec(spec).ephemeral(self.ephemeral);
            if let Some(ref c) = self.compiler { b = b.compiler(c); }
            b.build()?
        };
        let client = PawClient::new(&self.config);
        let program = client.compile(request).await?;
        let dir = client.download_paw(&program.id).await?;
        download_assets(&self.config, &dir).await?;

        cfg_backend_load!(self, dir)
    }
}

// ── Shared helpers ─────────────────────────────────────────────────

async fn download_assets(config: &PawConfig, program_dir: &std::path::Path) -> Result<(), Error> {
    let bundle = paw_core::PawBundle::load_from_dir(program_dir)?;
    let interpreter = bundle.interpreter_model();

    #[cfg(feature = "candle")]
    paw_candle::ensure_assets(config, program_dir, interpreter).await?;

    #[cfg(all(feature = "llamacpp", not(feature = "candle")))]
    ensure_gguf_cached(config, interpreter)?;

    Ok(())
}

#[cfg(all(feature = "llamacpp", not(feature = "candle")))]
fn ensure_gguf_cached(config: &PawConfig, interpreter: &str) -> Result<(), Error> {
    use paw_core::cache::known_models;
    let file_name = known_models::interpreter_to_gguf(interpreter)
        .map(|(_, f)| f)
        .ok_or_else(|| Error::UnsupportedModel(interpreter.to_string()))?;
    let gguf_path = config.base_models_dir().join(file_name);
    if !gguf_path.exists() {
        return Err(Error::Cache(format!(
            "GGUF not cached at {}. Run with candle backend first or download manually.",
            gguf_path.display()
        )));
    }
    Ok(())
}
