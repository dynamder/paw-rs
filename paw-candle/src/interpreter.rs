use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use paw_core::{Error, InterpreterModel};

use crate::models::{QuantizedModel, gpt2::Gpt2Model, qwen3::Qwen3Model};
use crate::pool;

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

pub struct SharedModelHandle {
    pub(crate) model: Arc<Mutex<Box<dyn QuantizedModel>>>,
    pub(crate) pool: Arc<pool::ModelPool>,
}

// ── Backend impl for Candle ────────────────────────────────────────────────

pub struct CandleBackend;

impl paw_core::Backend for CandleBackend {
    type SharedModel = SharedModelHandle;

    fn load_from_dir(dir: std::path::PathBuf) -> Result<Box<dyn paw_core::PawFnTrait>, Error> {
        use crate::{PawCandleConfig, runtime::PawFnLoader};
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
        let gguf_path = config.base_models_dir().join(T::GGUF_FILE);

        let (model, pool) = pool::get_or_load_model(T::INTERPRETER, 1, move || {
            if !gguf_path.exists() {
                return Err(Error::Cache(format!(
                    "GGUF not cached at {}",
                    gguf_path.display()
                )));
            }
            load_gguf::<T>(&gguf_path, &device)
                .map_err(|e| Error::Other(format!("model load: {e}")))
        })?;

        Ok(SharedModelHandle { model, pool })
    }

    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        handle: Self::SharedModel,
    ) -> Result<Box<dyn paw_core::PawFnTrait>, Error> {
        use crate::{PawCandleConfig, runtime::PawFnLoader};
        let inner = PawFnLoader::new(dir)
            .config(PawCandleConfig::default())
            .load_with_model(handle.pool, handle.model)?;
        Ok(Box::new(inner))
    }
}
