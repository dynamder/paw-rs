use std::marker::PhantomData;

use paw_candle::{
    get_or_load_model, DevicePreference, InterpreterModel, PawCandleConfig, PawFnLoader,
    PawFnTrait, PawFunction, PawRuntimeOptions,
};
use paw_core::{CompileRequest, PawClient, PawConfig, Error};

// ── State markers ─────────────────────────────────────────────────────

/// Initial builder state: no slug or spec set yet.
pub struct Unset;

/// Builder has a slug, can call `.load()`.
pub struct ForLoad;

/// Builder has a spec, can call `.compile()`.
pub struct ForCompile;

// ── PawFn ─────────────────────────────────────────────────────────────

/// Typed PAW function, parameterized by the interpreter model.
///
/// The builder ([`PawFnBuilder::builder()`]) returns `Box<dyn PawFnTrait>`.
/// For static typing and model sharing, use a concrete type:
///
/// ```rust,no_run
/// use paw_rs::prelude::*;
/// use paw_rs::paw_candle::Qwen3_0_6B;
/// # async fn example() -> std::result::Result<(), paw_core::Error> {
/// let mut f = PawFn::<Qwen3_0_6B>::load_slug("email-triage").await?;
/// # Ok(()) }
/// ```
pub struct PawFn<T: InterpreterModel = paw_candle::Dynamic> {
    inner: PawFunction,
    _phantom: PhantomData<T>,
}

impl<T: InterpreterModel> PawFn<T> {
    /// Wrap a pre-loaded `PawFunction` (low-level path).
    pub fn from_inner(inner: PawFunction) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Run inference with default options (greedy decoding, no token limit).
    pub fn run(&mut self, input: &str) -> Result<String, Error> {
        self.inner.run(input, &PawRuntimeOptions::default())
    }

    /// Run inference with custom [`PawRuntimeOptions`].
    pub fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.inner.run(input, opts)
    }

    /// The interpreter model identifier (e.g. `"Qwen/Qwen3-0.6B"`).
    pub fn interpreter(&self) -> &str {
        self.inner.interpreter()
    }

    // ── Static-typed constructors (with shared model) ──────────────

    /// Load an existing program by slug, reusing a cached base model
    /// identified by `T`.  Multiple [`PawFn<T>`] instances of the same
    /// `T` share a single base model in memory.
    pub async fn load_slug(slug: &str) -> Result<Self, Error> {
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
        let inner = PawFnLoader::new(dir)
            .config(candle_config)
            .load_with_model(shared)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Compile a spec and load it as a typed [`PawFn<T>`].
    ///
    /// The model `T` determines the compiler and enables base model sharing.
    pub async fn compile_spec(spec: &str, compiler: &str) -> Result<Self, Error> {
        let config = PawConfig::from_env();
        let client = PawClient::new(&config);
        let request = CompileRequest::builder()
            .spec(spec)
            .compiler(compiler)
            .ephemeral(true)
            .build()?;
        let program = client.compile(request).await?;
        let dir = client.download_paw(&program.id).await?;
        download_assets(&config, &dir).await?;
        let candle_config = PawCandleConfig::builder()
            .core(config.clone())
            .device(DevicePreference::Auto)
            .build();
        let device = paw_candle::runtime::select_device_for_loading(&candle_config)?;
        let shared = get_or_load_model::<T>(&config, &device)?;
        let inner = PawFnLoader::new(dir)
            .config(candle_config)
            .load_with_model(shared)?;
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }
}

impl<T: InterpreterModel> PawFnTrait for PawFn<T> {
    fn run(&mut self, input: &str) -> Result<String, Error> {
        self.run(input)
    }
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.run_with(input, opts)
    }
    fn interpreter(&self) -> &str {
        self.interpreter()
    }
}

// ── PawFnBuilder ──────────────────────────────────────────────────────

/// Type-state builder that returns `Box<dyn PawFnTrait>`.
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
    /// Start building a [`PawFn`]. Equivalent to [`PawFnBuilder::new`].
    pub fn builder() -> Self {
        Self::new()
    }

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
    pub async fn load(self) -> Result<Box<dyn PawFnTrait>, Error> {
        let slug = self.slug.expect("slug must be set in ForLoad state");
        let client = PawClient::new(&self.config);
        let program_id = client.resolve_slug(&slug).await?;
        let dir = client.download_paw(&program_id).await?;
        download_assets(&self.config, &dir).await?;
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::builder().core(self.config).device(self.device).build())
            .load()?;
        Ok(Box::new(inner))
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

    pub async fn compile(self) -> Result<Box<dyn PawFnTrait>, Error> {
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
        Ok(Box::new(inner))
    }
}

// ── Shared helpers ────────────────────────────────────────────────────

async fn download_assets(
    config: &PawConfig,
    program_dir: &std::path::Path,
) -> Result<(), Error> {
    let bundle = paw_core::PawBundle::load_from_dir(program_dir)?;
    paw_candle::ensure_assets(config, program_dir, bundle.interpreter_model()).await
}
