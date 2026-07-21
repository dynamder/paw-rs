use std::cell::RefCell;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Once;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use paw_core::{Error, PawBundle, PawFnTrait, PawRuntimeOptions};

use crate::config::PawLlamaCppConfig;

static INIT: Once = Once::new();
static mut BACKEND_PTR: *const LlamaBackend = std::ptr::null();

fn global_backend() -> &'static LlamaBackend {
    unsafe {
        INIT.call_once(|| {
            let mut backend = LlamaBackend::init().expect("failed to initialize llama.cpp backend");
            #[cfg(not(feature = "tracing"))]
            backend.void_logs();
            #[cfg(feature = "tracing")]
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
            BACKEND_PTR = Box::into_raw(Box::new(backend));
        });
        &*BACKEND_PTR
    }
}

fn eos_from_gguf(model: &LlamaModel) -> u32 {
    model
        .meta_val_str("tokenizer.ggml.eos_token_id")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

struct ModelBundle {
    model: LlamaModel,
}

#[allow(dead_code)]
pub struct PawFunction {
    bundle: ModelBundle,
    adapter: Option<LlamaLoraAdapter>,
    ctx: RefCell<Option<LlamaContext<'static>>>,
    n_ctx: usize,
    seed: u32,
    prefix_text: String,
    suffix_text: String,
    n_prefix: usize,
    prefix_evaluated: RefCell<bool>,
    eos_token_id: u32,
    interpreter: String,
}

impl Drop for PawFunction {
    fn drop(&mut self) {
        if let (Some(ctx), Some(adapter)) = (self.ctx.borrow_mut().as_mut(), self.adapter.as_mut())
        {
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

        let n_prefix = self.n_prefix;
        let mut guard = self.ctx.borrow_mut();
        let ctx = guard.as_mut().unwrap();

        if !*self.prefix_evaluated.borrow() {
            if n_prefix > 0 {
                let prefix_llama: Vec<LlamaToken> = all_tokens[..n_prefix].to_vec();
                let mut batch = LlamaBatch::new(n_prefix.max(1), 1);
                for (i, &t) in prefix_llama.iter().enumerate() {
                    batch
                        .add(t, i as i32, &[0], false)
                        .map_err(|e| Error::Other(format!("prefix add: {e}")))?;
                }
                ctx.decode(&mut batch)
                    .map_err(|e| Error::Other(format!("prefix eval: {e}")))?;
            }
            *self.prefix_evaluated.borrow_mut() = true;
            //info!("Prefix evaluated and cached ({} tokens)", n_prefix);
        }

        let _ = ctx.clear_kv_cache_seq(Some(0), Some(n_prefix as u32), None);

        let input_tokens: Vec<LlamaToken> = all_tokens[n_prefix..].to_vec();
        if !input_tokens.is_empty() {
            let mut batch = LlamaBatch::new(input_tokens.len().max(1), 1);
            let last = input_tokens.len() as i32 - 1;
            for (i, &t) in input_tokens.iter().enumerate() {
                let pos = n_prefix as i32 + i as i32;
                batch
                    .add(t, pos, &[0], i as i32 == last)
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

            if self.bundle.model.is_eog_token(token) || token.0 as u32 == self.eos_token_id {
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

impl PawFnTrait for PawFunction {
    fn run(&mut self, input: &str) -> Result<String, Error> {
        PawFunction::run(self, input, &PawRuntimeOptions::default())
    }
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        PawFunction::run(self, input, opts)
    }
    fn interpreter(&self) -> &str {
        &self.interpreter
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

    pub fn load(self) -> Result<Box<PawFunction>, Error> {
        let bundle = self.load_bundle()?;
        let (model, adapter) = self.load_model(&bundle)?;
        self.assemble_boxed(bundle, model, adapter)
    }

    fn load_bundle(&self) -> Result<PawBundle, Error> {
        PawBundle::load_from_dir(&self.program_dir)
    }

    fn load_model(
        &self,
        bundle: &PawBundle,
    ) -> Result<(LlamaModel, Option<LlamaLoraAdapter>), Error> {
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
        let n_layers = self.config.n_gpu_layers.max(0) as u32;
        let mp = if n_layers > 0 {
            LlamaModelParams::default().with_n_gpu_layers(n_layers)
        } else {
            LlamaModelParams::default()
        };
        let model = LlamaModel::load_from_file(global_backend(), &gguf_path, &mp)
            .map_err(|e| Error::Other(format!("model load: {e}")))?;
        //info!("Model loaded ({} params)", model.n_params());

        let adapter = if bundle.adapter_path.exists() {
            Some(
                model
                    .lora_adapter_init(&bundle.adapter_path)
                    .map_err(|e| Error::Other(format!("LoRA adapter load failed: {e}")))?,
            )
        } else {
            None
        };
        Ok((model, adapter))
    }

    fn assemble_boxed(
        &self,
        bundle: PawBundle,
        model: LlamaModel,
        adapter: Option<LlamaLoraAdapter>,
    ) -> Result<Box<PawFunction>, Error> {
        let n_ctx = self.config.core.n_ctx() as usize;
        let eos_token_id = eos_from_gguf(&model);
        let (prefix_text, suffix_text) = bundle.split_template();

        let prefix_tokens = model
            .str_to_token(&prefix_text, AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize prefix: {e}")))?;
        let n_prefix = prefix_tokens.len();

        // Create Box FIRST so model is on the heap
        let mut pf = Box::new(PawFunction {
            bundle: ModelBundle { model },
            adapter,
            ctx: RefCell::new(None),
            n_ctx,
            seed: self.config.seed,
            prefix_text,
            suffix_text,
            n_prefix,
            prefix_evaluated: RefCell::new(false),
            eos_token_id,
            interpreter: bundle.interpreter_model().to_string(),
        });

        // Create context from model already on the heap inside the Box
        let mut cp = LlamaContextParams::default().with_n_ctx(Some(
            NonZeroU32::new(n_ctx as u32).unwrap_or(NonZeroU32::new(2048).unwrap()),
        ));
        if let Some(t) = self.config.n_threads {
            cp = cp.with_n_threads(t);
        }
        if let Some(t) = self.config.n_threads_batch {
            cp = cp.with_n_threads_batch(t);
        }

        let ctx = pf
            .bundle
            .model
            .new_context(global_backend(), cp)
            .map_err(|e| Error::Other(format!("new_context: {e}")))?;

        if let Some(ref mut a) = pf.adapter {
            ctx.lora_adapter_set(a, 1.0)
                .map_err(|e| Error::Other(format!("lora set: {e}")))?;
            //info!("LoRA applied");
        }

        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        *pf.ctx.borrow_mut() = Some(ctx);

        // info!(
        //     "Program loaded: model={}, prefix={} tokens, eos={}{}",
        //     bundle.interpreter_model(),
        //     n_prefix,
        //     eos_token_id,
        //     if pf.adapter.is_some() {
        //         " (with LoRA)"
        //     } else {
        //         ""
        //     },
        // );
        Ok(pf)
    }
}
