use crate::error::Error;

// ── Runtime options ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PawRuntimeOptions {
    pub max_tokens: Option<usize>,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for PawRuntimeOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.0,
            top_p: 1.0,
        }
    }
}

// ── PawFnTrait (dynamic dispatch) ──────────────────────────────────────

pub trait PawFnTrait: Send {
    fn run(&mut self, input: &str) -> Result<String, Error>;
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error>;
    fn interpreter(&self) -> &str;
}

// ── InterpreterModel (model identifier) ────────────────────────────────

pub trait InterpreterModel: Send + Sync + 'static {
    const INTERPRETER: &'static str;
    const GGUF_FILE: &'static str;
}

pub struct Qwen3_0_6B;
impl InterpreterModel for Qwen3_0_6B {
    const INTERPRETER: &'static str = "Qwen/Qwen3-0.6B";
    const GGUF_FILE: &'static str = "qwen3-0.6b-q6_k.gguf";
}

pub struct Gpt2;
impl InterpreterModel for Gpt2 {
    const INTERPRETER: &'static str = "gpt2";
    const GGUF_FILE: &'static str = "gpt2-qs_0.gguf";
}

/// Sentinel for dynamic dispatch (interpreter unknown at compile time).
pub enum Dynamic {}
impl InterpreterModel for Dynamic {
    const INTERPRETER: &'static str = "(dynamic)";
    const GGUF_FILE: &'static str = "";
}

// ── Backend trait ──────────────────────────────────────────────────────

pub trait Backend: Send + Sync + 'static {
    /// Create a loaded PawFunction from a downloaded program directory.
    fn load_from_dir(dir: std::path::PathBuf) -> Result<Box<dyn PawFnTrait>, Error>
    where
        Self: Sized;

    /// Download/cache base model + tokenizer assets.
    fn ensure_assets(
        config: &crate::PawConfig,
        dir: &std::path::Path,
        interpreter: &str,
    ) -> impl std::future::Future<Output = Result<(), Error>>;

    /// Return a shared cached model handle for a given interpreter type.
    /// The handle is backend-opaque but can be passed back to a loader.
    type SharedModel: Send;
    fn get_or_load_model<T: InterpreterModel>(
        config: &crate::PawConfig,
    ) -> Result<Self::SharedModel, Error>;
    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        model: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error>;
}

// ── Backend marker types ───────────────────────────────────────────────

pub enum Candle {}
impl Backend for Candle {
    type SharedModel = (); // placeholder
    fn load_from_dir(_dir: std::path::PathBuf) -> Result<Box<dyn PawFnTrait>, Error> {
        error_no_backend("candle")
    }
    async fn ensure_assets(
        config: &crate::PawConfig,
        dir: &std::path::Path,
        _interpreter: &str,
    ) -> Result<(), Error> {
        let _ = config;
        let _ = dir;
        error_no_backend("candle")
    }
    fn get_or_load_model<T: InterpreterModel>(
        _config: &crate::PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        error_no_backend("candle")
    }
    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        _model: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error> {
        let _ = dir;
        error_no_backend("candle")
    }
}

pub enum LlamaCpp {}
impl Backend for LlamaCpp {
    type SharedModel = ();
    fn load_from_dir(dir: std::path::PathBuf) -> Result<Box<dyn PawFnTrait>, Error> {
        let _ = dir;
        error_no_backend("llamacpp")
    }
    async fn ensure_assets(
        config: &crate::PawConfig,
        dir: &std::path::Path,
        _interpreter: &str,
    ) -> Result<(), Error> {
        let _ = config;
        let _ = dir;
        error_no_backend("llamacpp")
    }
    fn get_or_load_model<T: InterpreterModel>(
        _config: &crate::PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        error_no_backend("llamacpp")
    }
    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        _model: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error> {
        let _ = dir;
        error_no_backend("llamacpp")
    }
}

pub enum DynamicBackend {}
impl Backend for DynamicBackend {
    type SharedModel = ();
    fn load_from_dir(_dir: std::path::PathBuf) -> Result<Box<dyn PawFnTrait>, Error> {
        Err(Error::Other("no backend selected".into()))
    }
    async fn ensure_assets(
        _config: &crate::PawConfig,
        _dir: &std::path::Path,
        _interpreter: &str,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn get_or_load_model<T: InterpreterModel>(
        _config: &crate::PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        Err(Error::Other("no backend selected".into()))
    }
    fn load_from_dir_with_model(
        _dir: std::path::PathBuf,
        _model: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error> {
        Err(Error::Other("no backend selected".into()))
    }
}

fn error_no_backend(name: &str) -> ! {
    panic!("Backend `{name}` is not compiled. Add `features = [\"{name}\"]` to paw-rs deps.")
}
