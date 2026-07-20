use std::marker::PhantomData;

use paw_core::{
    Backend, CompileRequest, Error, InterpreterModel, PawClient, PawConfig, PawFnTrait,
    PawRuntimeOptions,
};

// ── Default backend (selected by Cargo features) ─────────────────────

#[cfg(all(feature = "llamacpp", not(feature = "candle")))]
pub type DefaultBackend = paw_llamacpp::LlamaCppBackend;

#[cfg(feature = "candle")]
pub type DefaultBackend = paw_candle::interpreter::CandleBackend;

// ── State markers ─────────────────────────────────────────────────────

pub struct Unset;
pub struct ForLoad;
pub struct ForCompile;

// ── PawFn ─────────────────────────────────────────────────────────────

/// Typed PAW function, parameterized by model type `T` and backend `B`.
///
/// `T` determines which interpreter model to use (e.g. `Qwen3_0_6B`).
/// `B` selects the inference engine (e.g. `Candle`, `LlamaCpp`).
///
/// ```rust,no_run
/// use paw_core::{Candle, Qwen3_0_6B};
/// use paw_rs::prelude::*;
/// # async fn example() -> Result<(), paw_core::Error> {
/// let mut f = PawFn::<Qwen3_0_6B, Candle>::load_slug("email-triage").await?;
/// println!("{}", f.run("Help!")?);
/// # Ok(()) }
/// ```
pub struct PawFn<T: InterpreterModel, B: Backend> {
    inner: Box<dyn PawFnTrait>,
    _model: PhantomData<T>,
    _backend: PhantomData<B>,
}

impl<T: InterpreterModel, B: Backend> PawFn<T, B> {
    pub fn run(&mut self, input: &str) -> Result<String, Error> {
        self.inner.run(input)
    }

    pub fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.inner.run_with(input, opts)
    }

    pub fn interpreter(&self) -> &str {
        self.inner.interpreter()
    }

    /// Load an existing program by slug, using a shared base model.
    pub async fn load_slug(slug: &str) -> Result<Self, Error> {
        let config = PawConfig::from_env();
        let client = PawClient::new(&config);
        let program_id = client.resolve_slug(slug).await?;
        let dir = client.download_paw(&program_id).await?;
        B::ensure_assets(&config, &dir, T::INTERPRETER).await?;
        let model_handle = B::get_or_load_model::<T>(&config)?;
        let inner = B::load_from_dir_with_model(dir, model_handle)?;
        Ok(Self {
            inner,
            _model: PhantomData,
            _backend: PhantomData,
        })
    }

    /// Compile a spec and load as a typed [`PawFn`].
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
        B::ensure_assets(&config, &dir, T::INTERPRETER).await?;
        let model_handle = B::get_or_load_model::<T>(&config)?;
        let inner = B::load_from_dir_with_model(dir, model_handle)?;
        Ok(Self {
            inner,
            _model: PhantomData,
            _backend: PhantomData,
        })
    }
}

impl<T: InterpreterModel, B: Backend> PawFnTrait for PawFn<T, B> {
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
    pub fn config(mut self, config: PawConfig) -> Self {
        self.config = config;
        self
    }
}

impl PawFnBuilder<Unset> {
    pub fn builder() -> Self {
        Self::default()
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }
    pub fn ephemeral(mut self, e: bool) -> Self {
        self.ephemeral = e;
        self
    }
    pub fn slug(self, slug: impl Into<String>) -> PawFnBuilder<ForLoad> {
        PawFnBuilder {
            config: self.config,
            slug: Some(slug.into()),
            program_id: None,
            spec: None,
            compiler: self.compiler,
            ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
    pub fn id(self, program_id: impl Into<String>) -> PawFnBuilder<ForLoad> {
        PawFnBuilder {
            config: self.config,
            program_id: Some(program_id.into()),
            slug: None,
            spec: None,
            compiler: self.compiler,
            ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
    pub fn spec(self, spec: impl Into<String>) -> PawFnBuilder<ForCompile> {
        PawFnBuilder {
            config: self.config,
            slug: None,
            program_id: None,
            spec: Some(spec.into()),
            compiler: self.compiler,
            ephemeral: self.ephemeral,
            _marker: PhantomData,
        }
    }
}

impl Default for PawFnBuilder<Unset> {
    fn default() -> Self {
        Self {
            config: PawConfig::from_env(),
            slug: None,
            program_id: None,
            spec: None,
            compiler: None,
            ephemeral: false,
            _marker: PhantomData,
        }
    }
}

// ── Load mode ──────────────────────────────────────────────────────

impl PawFnBuilder<ForLoad> {
    pub async fn load(self) -> Result<Box<dyn PawFnTrait>, Error> {
        let config = self.config;
        let client = PawClient::new(&config);
        let program_id = match (self.slug.as_deref(), self.program_id.as_deref()) {
            (Some(slug), _) => client.resolve_slug(slug).await?,
            (_, Some(id)) => id.to_string(),
            (None, None) => {
                return Err(Error::Other(
                    "requires .slug() or .id() before .load()".into(),
                ))
            }
        };
        let dir = client.download_paw(&program_id).await?;
        download_assets::<DefaultBackend>(&config, &dir).await?;
        DefaultBackend::load_from_dir(dir)
    }
}

// ── Compile mode ──────────────────────────────────────────────────

impl PawFnBuilder<ForCompile> {
    pub fn compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }
    pub fn ephemeral(mut self, e: bool) -> Self {
        self.ephemeral = e;
        self
    }
    pub async fn compile(self) -> Result<Box<dyn PawFnTrait>, Error> {
        let spec = self.spec.expect("spec must be set for ForCompile");
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
        download_assets::<DefaultBackend>(&self.config, &dir).await?;
        DefaultBackend::load_from_dir(dir)
    }
}

// ── Shared helpers ─────────────────────────────────────────────────

async fn download_assets<B: Backend>(
    config: &PawConfig,
    program_dir: &std::path::Path,
) -> Result<(), Error> {
    let bundle = paw_core::PawBundle::load_from_dir(program_dir)?;
    B::ensure_assets(config, program_dir, bundle.interpreter_model()).await
}
