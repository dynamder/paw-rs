use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::Device;
use paw_core::{Error, InterpreterModel};

use crate::lora::GgufLoraAdapter;
use crate::models::{gpt2::Gpt2Model, qwen3::Qwen3Model, QuantizedModel};

/// Load a GGUF model for the given interpreter type.
fn load_gguf<T: InterpreterModel>(
    path: &Path,
    device: &Device,
) -> std::result::Result<Box<dyn QuantizedModel>, candle_core::Error> {
    let model: Box<dyn QuantizedModel> = match T::INTERPRETER {
        "Qwen/Qwen3-0.6B" => Box::new(Qwen3Model::from_gguf(path, device)?),
        "gpt2" => Box::new(Gpt2Model::from_gguf(path, device)?),
        _ => panic!("unknown interpreter: {}", T::INTERPRETER),
    };
    Ok(model)
}

fn cached_model_lock<T: InterpreterModel>() -> &'static OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> {
    static CACHE: OnceLock<Arc<Mutex<Box<dyn QuantizedModel>>>> = OnceLock::new();
    &CACHE
}

pub fn get_or_load_model<T: InterpreterModel>(
    config: &paw_core::PawConfig,
    device: &Device,
) -> paw_core::Result<Arc<Mutex<Box<dyn QuantizedModel>>>> {
    let cache = cached_model_lock::<T>();
    let shared = cache.get_or_init(|| {
        let gguf_path = config.base_models_dir().join(T::GGUF_FILE);
        let model = load_gguf::<T>(&gguf_path, device)
            .unwrap_or_else(|e| panic!("failed to load GGUF model {}: {e}", T::INTERPRETER));
        Arc::new(Mutex::new(model))
    });
    Ok(Arc::clone(shared))
}

// ── Backend impl for Candle ────────────────────────────────────────────────

pub struct CandleBackend;

impl paw_core::Backend for CandleBackend {
    type SharedModel = ();
    fn load_from_dir(dir: std::path::PathBuf) -> Result<Box<dyn paw_core::PawFnTrait>, Error> {
        use crate::{runtime::PawFnLoader, PawCandleConfig};
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::default())
            .load()?;
        Ok(Box::new(inner))
    }
    async fn ensure_assets(
        config: &paw_core::PawConfig,
        dir: &std::path::Path,
        interpreter: &str,
    ) -> Result<(), Error> {
        crate::runtime::ensure_assets(config, dir, interpreter).await
    }
    fn get_or_load_model<T: InterpreterModel>(
        config: &paw_core::PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        let device = Device::Cpu;
        get_or_load_model::<T>(config, &device)?;
        Ok(())
    }
    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        _model: Self::SharedModel,
    ) -> Result<Box<dyn paw_core::PawFnTrait>, Error> {
        use crate::{runtime::PawFnLoader, PawCandleConfig};
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::default())
            .load()?;
        Ok(Box::new(inner))
    }
}
