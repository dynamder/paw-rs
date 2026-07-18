use std::cell::RefCell;
use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use paw_core::{Error, PawBundle};
use tracing::info;

use crate::config::PawLlamaCppConfig;

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

struct ModelBundle {
    model: LlamaModel,
    backend: LlamaBackend,
}

pub struct PawFunction {
    bundle: ModelBundle,
    adapter: RefCell<Option<LlamaLoraAdapter>>,
    ctx: RefCell<Option<LlamaContext<'static>>>,

    n_ctx: usize,
    seed: u32,
    prefix_text: String,
    suffix_text: String,
    prefix_tokens: Vec<LlamaToken>,
    prefix_evaluated: RefCell<bool>,
}

impl Drop for PawFunction {
    fn drop(&mut self) {
        if let (Some(ctx), Some(adapter)) = (
            self.ctx.borrow_mut().as_mut(),
            self.adapter.borrow_mut().as_mut(),
        ) {
            let _ = ctx.lora_adapter_remove(adapter);
        }
        self.ctx.borrow_mut().take();
    }
}

unsafe impl Send for PawFunction {}
unsafe impl Sync for PawFunction {}

impl PawFunction {
    pub fn run(&self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        let full_prompt = format!("{}{}{}", self.prefix_text, input, self.suffix_text);

        let all_tokens = self
            .bundle
            .model
            .str_to_token(&full_prompt, AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        if all_tokens.len() >= self.n_ctx {
            return Err(Error::Other(format!(
                "Input too long: {} tokens (max {})",
                all_tokens.len(),
                self.n_ctx
            )));
        }

        let gen_limit = opts
            .max_tokens
            .map(|m| m.min(self.n_ctx - all_tokens.len()))
            .unwrap_or(self.n_ctx - all_tokens.len());

        let mut guard = self.ctx.borrow_mut();
        let ctx = guard.as_mut().unwrap();
        let n_prefix = self.prefix_tokens.len();

        if !*self.prefix_evaluated.borrow() {
            if n_prefix > 0 {
                let mut batch = LlamaBatch::new(n_prefix.max(1), 1);
                for (i, &t) in self.prefix_tokens.iter().enumerate() {
                    batch
                        .add(t, i as i32, &[0], false)
                        .map_err(|e| Error::Other(format!("prefix add: {e}")))?;
                }
                ctx.decode(&mut batch)
                    .map_err(|e| Error::Other(format!("prefix eval: {e}")))?;
            }
            *self.prefix_evaluated.borrow_mut() = true;
            info!("Prefix evaluated and cached ({} tokens)", n_prefix);
        }

        let _ = ctx.clear_kv_cache_seq(Some(0), Some(n_prefix as u32), None);

        let input_tokens: Vec<LlamaToken> = all_tokens[n_prefix..].to_vec();
        if !input_tokens.is_empty() {
            let mut batch = LlamaBatch::new(input_tokens.len().max(1), 1);
            let last = input_tokens.len() as i32 - 1;
            for (i, &t) in input_tokens.iter().enumerate() {
                let pos = n_prefix as i32 + i as i32;
                let is_last = i as i32 == last;
                batch
                    .add(t, pos, &[0], is_last)
                    .map_err(|e| Error::Other(format!("input add: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| Error::Other(format!("input eval: {e}")))?;
        }

        let mut pos = all_tokens.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        for _ in 0..gen_limit {
            if pos >= self.n_ctx as i32 {
                break;
            }

            let mut data = ctx.token_data_array();
            if opts.temperature > 0.0 {
                LlamaSampler::temp(opts.temperature as f32).apply(&mut data);
                LlamaSampler::top_p(opts.top_p as f32, 1).apply(&mut data);
                LlamaSampler::dist(self.seed).apply(&mut data);
            } else {
                LlamaSampler::greedy().apply(&mut data);
            }
            let token = match data.selected_token() {
                Some(t) => t,
                None => break,
            };

            if self.bundle.model.is_eog_token(token) {
                break;
            }

            if let Ok(piece) = self
                .bundle
                .model
                .token_to_piece(token, &mut decoder, false, None)
            {
                output.push_str(&piece);
            }

            let mut single = LlamaBatch::new(1, 1);
            single
                .add(token, pos, &[0], true)
                .map_err(|e| Error::Other(format!("decode add: {e}")))?;
            ctx.decode(&mut single)
                .map_err(|e| Error::Other(format!("decode: {e}")))?;
            pos += 1;
        }

        Ok(output.trim().to_string())
    }
}

pub struct PawFnLoader {
    program_dir: PathBuf,
    config: PawLlamaCppConfig,
}

impl PawFnLoader {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            program_dir: dir.into(),
            config: PawLlamaCppConfig::default(),
        }
    }

    pub fn config(mut self, config: PawLlamaCppConfig) -> Self {
        self.config = config;
        self
    }

    pub fn load(self) -> Result<PawFunction, Error> {
        let bundle = self.load_bundle()?;
        let (backend, model, adapter) = self.load_model(&bundle)?;
        self.assemble(bundle, backend, model, adapter)
    }

    fn load_bundle(&self) -> Result<PawBundle, Error> {
        PawBundle::load_from_dir(&self.program_dir)
    }

    fn load_model(
        &self,
        bundle: &PawBundle,
    ) -> Result<(LlamaBackend, LlamaModel, Option<LlamaLoraAdapter>), Error> {
        use paw_core::cache::known_models;
        let model_name = bundle.interpreter_model();
        let filename = match model_name {
            "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => known_models::QWEN3_0_6B_GGUF_FILE,
            "gpt2" | "gpt2-q8_0" => known_models::GPT2_GGUF_FILE,
            _ => return Err(Error::UnsupportedModel(model_name.to_string())),
        };
        let gguf_path = self.config.core.base_models_dir().join(filename);
        if !gguf_path.exists() {
            return Err(Error::Cache(format!(
                "GGUF not cached at {}",
                gguf_path.display()
            )));
        }
        info!("Loading GGUF: {}", gguf_path.display());
        let backend = LlamaBackend::init().map_err(|e| Error::Other(format!("backend: {e}")))?;
        let n_layers = self.config.n_gpu_layers.max(0) as u32;
        let mp = if n_layers > 0 {
            LlamaModelParams::default().with_n_gpu_layers(n_layers)
        } else {
            LlamaModelParams::default()
        };
        let model = LlamaModel::load_from_file(&backend, &gguf_path, &mp)
            .map_err(|e| Error::Other(format!("model load: {e}")))?;
        info!("Model loaded ({} params)", model.n_params());

        let adapter = if bundle.adapter_path.exists() {
            info!("Loading LoRA adapter: {}", bundle.adapter_path.display());
            match model.lora_adapter_init(&bundle.adapter_path) {
                Ok(a) => {
                    info!("LoRA adapter loaded");
                    Some(a)
                }
                Err(e) => {
                    tracing::warn!("LoRA adapter load failed: {e}");
                    None
                }
            }
        } else {
            None
        };
        Ok((backend, model, adapter))
    }

    fn assemble(
        &self,
        bundle: PawBundle,
        backend: LlamaBackend,
        model: LlamaModel,
        mut adapter: Option<LlamaLoraAdapter>,
    ) -> Result<PawFunction, Error> {
        let n_ctx = self.config.core.n_ctx() as usize;
        let (prefix_text, suffix_text) = bundle.split_template();

        let prefix_tokens = model
            .str_to_token(&prefix_text, AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize prefix: {e}")))?;

        let mut cp = LlamaContextParams::default().with_n_ctx(Some(
            NonZeroU32::new(n_ctx as u32).unwrap_or(NonZeroU32::new(2048).unwrap()),
        ));
        if let Some(t) = self.config.n_threads {
            cp = cp.with_n_threads(t);
        }
        if let Some(t) = self.config.n_threads_batch {
            cp = cp.with_n_threads_batch(t);
        }

        let ctx = model
            .new_context(&backend, cp)
            .map_err(|e| Error::Other(format!("new_context: {e}")))?;

        if let Some(ref mut a) = adapter {
            ctx.lora_adapter_set(a, 1.0)
                .map_err(|e| Error::Other(format!("lora set: {e}")))?;
            info!("LoRA applied");
        }

        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        info!(
            "Program loaded: model={}, prefix={} tokens{}",
            bundle.interpreter_model(),
            prefix_tokens.len(),
            if adapter.is_some() {
                " (with LoRA)"
            } else {
                ""
            },
        );
        Ok(PawFunction {
            bundle: ModelBundle { model, backend },
            adapter: RefCell::new(adapter),
            ctx: RefCell::new(Some(ctx)),
            n_ctx,
            seed: self.config.seed,
            prefix_text,
            suffix_text,
            prefix_tokens,
            prefix_evaluated: RefCell::new(false),
        })
    }
}
