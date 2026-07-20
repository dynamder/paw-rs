use paw_core::{Backend, Error, InterpreterModel, PawFnTrait, PawConfig};

pub struct LlamaCppBackend;

impl Backend for LlamaCppBackend {
    type SharedModel = ();

    fn load_from_dir(dir: std::path::PathBuf) -> Result<Box<dyn PawFnTrait>, Error> {
        use crate::{PawLlamaCppConfig, PawFnLoader};
        let loader = PawFnLoader::new(dir)
            .config(PawLlamaCppConfig::default())
            .load()?;
        let result: Box<dyn PawFnTrait> = loader;
        Ok(result)
    }

    async fn ensure_assets(
        _config: &PawConfig,
        _dir: &std::path::Path,
        interpreter: &str,
    ) -> Result<(), Error> {
        use paw_core::cache::known_models;
        let file_name = known_models::interpreter_to_gguf(interpreter)
            .map(|(_, f)| f)
            .ok_or_else(|| Error::UnsupportedModel(interpreter.to_string()))?;
        let gguf_path = _config.base_models_dir().join(file_name);
        if !gguf_path.exists() {
            return Err(Error::Cache(format!(
                "GGUF not cached at {}. Run with candle backend first or download manually.",
                gguf_path.display()
            )));
        }
        Ok(())
    }

    fn get_or_load_model<T: InterpreterModel>(
        _config: &PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        Ok(())
    }

    fn load_from_dir_with_model(
        dir: std::path::PathBuf,
        _model: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error> {
        Self::load_from_dir(dir)
    }
}
