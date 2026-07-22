use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use paw_core::{Backend, Error, InterpreterModel, PawConfig, PawFnTrait};

use crate::pool::{self, ModelPool};
use crate::{PawFnLoader, PawLlamaCppConfig};

pub struct LlamaCppBackend;

pub struct SharedModelHandle {
    pub(crate) model: Arc<LlamaModel>,
    pub(crate) pool: Arc<ModelPool>,
}

impl Backend for LlamaCppBackend {
    type SharedModel = SharedModelHandle;

    fn load_from_dir(dir: PathBuf) -> Result<Box<dyn PawFnTrait>, Error> {
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
        config: &PawConfig,
    ) -> Result<Self::SharedModel, Error> {
        // Initialize llama backend once
        static BACKEND_INIT: std::sync::Once = std::sync::Once::new();
        static mut BACKEND_PTR: *const LlamaBackend = std::ptr::null();
        let backend = unsafe {
            BACKEND_INIT.call_once(|| {
                let mut b = LlamaBackend::init().expect("failed to init llama backend");
                #[cfg(not(feature = "tracing"))]
                b.void_logs();
                #[cfg(feature = "tracing")]
                llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
                BACKEND_PTR = Box::into_raw(Box::new(b));
            });
            &*BACKEND_PTR
        };

        use paw_core::cache::known_models;
        let filename = match T::INTERPRETER {
            "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => known_models::QWEN3_0_6B_GGUF_FILE,
            "gpt2" | "gpt2-q8_0" => known_models::GPT2_GGUF_FILE,
            _ => return Err(Error::UnsupportedModel(T::INTERPRETER.to_string())),
        };
        let gguf_path = config.base_models_dir().join(filename);
        if !gguf_path.exists() {
            return Err(Error::Cache(format!(
                "GGUF not cached at {}",
                gguf_path.display()
            )));
        }

        let gguf_path_clone = gguf_path.clone();
        let (model, pool) = pool::get_or_load_model(T::INTERPRETER, 1, move || {
            LlamaModel::load_from_file(backend, &gguf_path_clone, &Default::default())
                .map_err(|e| Error::Other(format!("model load: {e}")))
        })?;

        Ok(SharedModelHandle { model, pool })
    }

    fn load_from_dir_with_model(
        dir: PathBuf,
        handle: Self::SharedModel,
    ) -> Result<Box<dyn PawFnTrait>, Error> {
        use std::cell::RefCell;
        use std::num::NonZeroU32;
        use llama_cpp_2::context::{LlamaContext, params::LlamaContextParams};
        use llama_cpp_2::model::AddBos;
        use paw_core::PawBundle;

        let bundle = PawBundle::load_from_dir(&dir)?;
        let config = PawLlamaCppConfig::default();
        let n_ctx = config.core.n_ctx() as usize;

        let eos_token_id = handle
            .model
            .meta_val_str("tokenizer.ggml.eos_token_id")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let (prefix_text, suffix_text) = bundle.split_template();
        let prefix_tokens = handle
            .model
            .str_to_token(&prefix_text, AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize prefix: {e}")))?;
        let n_prefix = prefix_tokens.len();

        let adapter = if bundle.adapter_path.exists() {
            Some(
                handle
                    .model
                    .lora_adapter_init(&bundle.adapter_path)
                    .map_err(|e| Error::Other(format!("LoRA adapter load failed: {e}")))?,
            )
        } else {
            None
        };

        let mut pf = Box::new(crate::runtime::PawFunction {
            model: handle.model,
            pool: handle.pool,
            adapter,
            ctx: RefCell::new(None),
            n_ctx,
            seed: config.seed,
            prefix_text,
            suffix_text,
            n_prefix,
            prefix_evaluated: RefCell::new(false),
            eos_token_id,
            interpreter: bundle.interpreter_model().to_string(),
        });

        static BACKEND_INIT2: std::sync::Once = std::sync::Once::new();
        static mut BACKEND_PTR2: *const LlamaBackend = std::ptr::null();
        let backend = unsafe {
            BACKEND_INIT2.call_once(|| {
                let b = LlamaBackend::init().expect("failed to init llama backend");
                BACKEND_PTR2 = Box::into_raw(Box::new(b));
            });
            &*BACKEND_PTR2
        };

        let cp = LlamaContextParams::default().with_n_ctx(Some(
            NonZeroU32::new(n_ctx as u32).unwrap_or(NonZeroU32::new(2048).unwrap()),
        ));

        let ctx = pf
            .model
            .new_context(backend, cp)
            .map_err(|e| Error::Other(format!("new_context: {e}")))?;

        if let Some(ref mut a) = pf.adapter {
            ctx.lora_adapter_set(a, 1.0)
                .map_err(|e| Error::Other(format!("lora set: {e}")))?;
        }

        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        *pf.ctx.borrow_mut() = Some(ctx);

        let result: Box<dyn PawFnTrait> = pf;
        Ok(result)
    }
}
