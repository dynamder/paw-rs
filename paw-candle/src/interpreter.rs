use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::Device;

use crate::models::QuantizedModel;

pub trait InterpreterModel: Send + Sync + 'static {
    const INTERPRETER: &'static str;
    const GGUF_FILE: &'static str;

    fn load_gguf(
        path: &Path,
        device: &Device,
    ) -> std::result::Result<Box<dyn QuantizedModel>, candle_core::Error>;

    #[doc(hidden)]
    fn cached_model() -> &'static OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>>;
}

pub fn get_or_load_model<T: InterpreterModel>(
    config: &paw_core::PawConfig,
    device: &Device,
) -> paw_core::Result<Arc<Mutex<Box<dyn QuantizedModel>>>> {
    let cache = T::cached_model();
    let shared = cache.get_or_init(|| {
        let gguf_path = config.base_models_dir().join(T::GGUF_FILE);
        let model = T::load_gguf(&gguf_path, device)
            .unwrap_or_else(|e| panic!("failed to load GGUF model {}: {e}", T::INTERPRETER));
        Arc::new(Mutex::new(model))
    });
    Ok(Arc::clone(shared))
}

pub struct Qwen3_0_6B;
impl InterpreterModel for Qwen3_0_6B {
    const INTERPRETER: &'static str = "Qwen/Qwen3-0.6B";
    const GGUF_FILE: &'static str = "qwen3-0.6b-q6_k.gguf";

    fn load_gguf(
        path: &Path,
        device: &Device,
    ) -> std::result::Result<Box<dyn QuantizedModel>, candle_core::Error> {
        let model = crate::models::qwen3::Qwen3Model::from_gguf(path, device)?;
        Ok(Box::new(model))
    }

    fn cached_model() -> &'static OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> {
        static CACHE: OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> = OnceLock::new();
        &CACHE
    }
}

/// Sentinel type for dynamic dispatch when the interpreter is
/// not known at compile time.  Used as the default type parameter
/// for `PawFn<T>`, enabling `PawFn::builder()` without turbofish.
pub enum Dynamic {}
impl InterpreterModel for Dynamic {
    const INTERPRETER: &'static str = "(dynamic)";
    const GGUF_FILE: &'static str = "";
    fn load_gguf(
        _path: &Path,
        _device: &Device,
    ) -> std::result::Result<Box<dyn QuantizedModel>, candle_core::Error> {
        unreachable!("Dynamic cannot load a model — use a concrete InterpreterModel type")
    }
    fn cached_model() -> &'static OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> {
        static CACHE: OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> = OnceLock::new();
        &CACHE
    }
}

pub struct Gpt2;
impl InterpreterModel for Gpt2 {
    const INTERPRETER: &'static str = "gpt2";
    const GGUF_FILE: &'static str = "gpt2-q8_0.gguf";

    fn load_gguf(
        path: &Path,
        device: &Device,
    ) -> std::result::Result<Box<dyn QuantizedModel>, candle_core::Error> {
        let model = crate::models::gpt2::Gpt2Model::from_gguf(path, device)?;
        Ok(Box::new(model))
    }

    fn cached_model() -> &'static OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> {
        static CACHE: OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> = OnceLock::new();
        &CACHE
    }
}
