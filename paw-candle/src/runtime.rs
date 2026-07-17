use std::path::PathBuf;

use candle_core::{Device, Tensor};
use paw_core::{Error, PawBundle};
use tracing::{debug, info};

use crate::config::{DevicePreference, PawCandleConfig};
use crate::kv_cache::PrefixKvCache;
use crate::lora::GgufLoraAdapter;
use crate::models::{gpt2::Gpt2Model, qwen3::Qwen3Model, QuantizedModel};
use crate::tokenizer::Tokenizer;

// ── Runtime options ────────────────────────────────────────────────────

/// Sampling and generation parameters for inference.
#[derive(Debug, Clone)]
pub struct PawRuntimeOptions {
    /// Maximum tokens to generate (`None` = up to context limit).
    pub max_tokens: Option<usize>,
    /// Sampling temperature (`0.0` = greedy decoding).
    pub temperature: f64,
    /// Top-p nucleus sampling (`1.0` = disabled).
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

// ── PawFunction (inference runtime) ────────────────────────────────────

/// The inference runtime for a PAW program.
///
/// Created by [`PawFnLoader`] — this struct only owns loaded state and
/// exposes [`run()`](PawFunction::run).
pub struct PawFunction {
    model: Box<dyn QuantizedModel>,
    tokenizer: Tokenizer,
    prefix_tokens: Vec<u32>,
    suffix_text: String,
    n_prefix: usize,
    n_ctx: usize,
    #[allow(dead_code)]
    lora_adapter: Option<GgufLoraAdapter>,
    #[allow(dead_code)]
    kv_cache: PrefixKvCache,
}

impl PawFunction {
    pub(crate) fn new(
        model: Box<dyn QuantizedModel>,
        tokenizer: Tokenizer,
        prefix_tokens: Vec<u32>,
        suffix_text: String,
        n_prefix: usize,
        n_ctx: usize,
        lora_adapter: Option<GgufLoraAdapter>,
        kv_cache: PrefixKvCache,
    ) -> Self {
        Self {
            model,
            tokenizer,
            prefix_tokens,
            suffix_text,
            n_prefix,
            n_ctx,
            lora_adapter,
            kv_cache,
        }
    }

    /// Run inference on the given input text.
    pub fn run(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        debug!("Running inference: input={}", &input[..input.len().min(60)]);

        let full_input = format!("{}{}", input, self.suffix_text);
        let input_tokens = self.tokenizer.encode(&full_input)?;

        let tokens_used = self.n_prefix + input_tokens.len();
        if tokens_used >= self.n_ctx {
            return Err(Error::Other(format!(
                "Input too long: {tokens_used} tokens (prefix={}, input={}), context={}",
                self.n_prefix,
                input_tokens.len(),
                self.n_ctx
            )));
        }

        let gen_limit = opts
            .max_tokens
            .map(|m| m.min(self.n_ctx - tokens_used))
            .unwrap_or(self.n_ctx - tokens_used);

        let mut token_ids = self.prefix_tokens.clone();
        token_ids.extend(&input_tokens);
        let start_len = token_ids.len();
        let device = self.model.device().clone();

        let te = |e: candle_core::Error| Error::Other(format!("tensor op: {e}"));

        for step in 0..gen_limit {
            let input = Tensor::new(&token_ids[..], &device)
                .map_err(&te)?
                .unsqueeze(0)
                .map_err(&te)?;

            let logits = self
                .model
                .forward(&input, 0)
                .map_err(|e| Error::Other(format!("forward: {e}")))?;

            let seq_len = logits.dim(1).map_err(&te)?;
            let last_logits = logits
                .squeeze(0)
                .map_err(&te)?
                .get(seq_len - 1)
                .map_err(&te)?;

            let next_id = last_logits
                .argmax(0)
                .map_err(&te)?
                .to_scalar::<u32>()
                .map_err(&te)?;

            if next_id == self.model.eos_token_id() {
                debug!("EOS token at step {step}");
                break;
            }

            token_ids.push(next_id);

            if step % 10 == 0 {
                debug!("step {step}: generated token {next_id}");
            }
        }

        let generated = &token_ids[start_len..];
        let output = self.tokenizer.decode(generated)?;
        debug!("Generated {} tokens", generated.len());
        Ok(output)
    }
}

// ── PawFnLoader (pure I/O) ─────────────────────────────────────────────

/// Local loader for PAW program bundles.
///
/// Assumes the program has already been downloaded — this struct only
/// handles **local file I/O**: parsing the bundle, loading the GGUF model,
/// loading the tokenizer, and assembling a [`PawFunction`].
///
/// # Example
///
/// ```rust,no_run
/// use paw_candle::{PawCandleConfig, PawFnLoader, PawRuntimeOptions};
///
/// # fn example() -> Result<(), paw_candle::Error> {
/// let config = PawCandleConfig::default();
/// let mut func = PawFnLoader::new("/path/to/program_dir")
///     .config(config)
///     .load()?;
/// let result = func.run("input", &PawRuntimeOptions::default())?;
/// # Ok(())
/// # }
/// ```
pub struct PawFnLoader {
    program_dir: PathBuf,
    config: PawCandleConfig,
}

impl PawFnLoader {
    /// Create a loader from a local program directory.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            program_dir: dir.into(),
            config: PawCandleConfig::default(),
        }
    }

    /// Bind a configuration.
    pub fn config(mut self, config: PawCandleConfig) -> Self {
        self.config = config;
        self
    }

    /// One-shot: parse bundle, load model, assemble function.
    pub fn load(self) -> Result<PawFunction, Error> {
        let bundle = self.load_bundle()?;
        let device = select_device(&self.config)?;
        let model = load_model(&bundle, &self.config, &device)?;
        let tokenizer = Tokenizer::new(&bundle)?;
        self.assemble(bundle, model, tokenizer)
    }

    // ── Fine-grained steps ────────────────────────────────────────────

    /// Parse the bundle directory into a [`PawBundle`].
    pub fn load_bundle(&self) -> Result<PawBundle, Error> {
        PawBundle::load_from_dir(&self.program_dir)
    }

    /// Load the GGUF quantized model described by the bundle.
    pub fn load_model(&self, bundle: &PawBundle) -> Result<Box<dyn QuantizedModel>, Error> {
        let device = select_device(&self.config)?;
        load_model(bundle, &self.config, &device)
    }

    /// Load the tokenizer from the bundle directory.
    pub fn load_tokenizer(&self, bundle: &PawBundle) -> Result<Tokenizer, Error> {
        Tokenizer::new(bundle)
    }

    /// Assemble all parts into a [`PawFunction`].
    pub fn assemble(
        &self,
        bundle: PawBundle,
        mut model: Box<dyn QuantizedModel>,
        tokenizer: Tokenizer,
    ) -> Result<PawFunction, Error> {
        let device = model.device();
        let lora = GgufLoraAdapter::from_gguf_file(&bundle.adapter_path, device).ok();
        if let Some(ref lora) = lora {
            let matched = model.set_lora(lora);
            tracing::info!("LoRA applied: {matched} weight matrices matched");
        }
        let (prefix_text, suffix_text) = bundle.split_template();
        let prefix_tokens = tokenizer.encode(&prefix_text)?;
        let n_prefix = prefix_tokens.len();
        let n_ctx = self.config.core.n_ctx() as usize;

        let kv_cache = PrefixKvCache::new(
            bundle.program_dir.join("prefix_kv_cache.bin"),
            model.num_layers(),
            model.head_dim(),
            model.num_kv_heads(),
            n_prefix,
            &model.device(),
        );

        info!(
            "Loaded program: {} prefix tokens, model={}",
            n_prefix,
            bundle.interpreter_model()
        );

        Ok(PawFunction::new(
            model,
            tokenizer,
            prefix_tokens,
            suffix_text,
            n_prefix,
            n_ctx,
            lora,
            kv_cache,
        ))
    }
}

// ── Private helpers ─────────────────────────────────────────────────

fn select_device(config: &PawCandleConfig) -> Result<Device, Error> {
    match config.device {
        DevicePreference::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(d) = Device::new_cuda(0) {
                info!("Using CUDA device");
                return Ok(d);
            }
            #[cfg(feature = "metal")]
            if let Ok(d) = Device::new_metal(0) {
                info!("Using Metal device");
                return Ok(d);
            }
            info!("Using CPU device");
            Ok(Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        DevicePreference::Cuda => Device::new_cuda(0).map_err(|e| Error::Other(e.to_string())),
        #[cfg(feature = "metal")]
        DevicePreference::Metal => Device::new_metal(0).map_err(|e| Error::Other(e.to_string())),
        DevicePreference::Cpu => Ok(Device::Cpu),
    }
}

fn load_model(
    bundle: &PawBundle,
    config: &PawCandleConfig,
    device: &Device,
) -> Result<Box<dyn QuantizedModel>, Error> {
    use paw_core::cache::known_models;

    let model_name = bundle.interpreter_model();

    let (_, filename) = if let (Some(r), Some(f)) = (&config.base_model_repo, &config.gguf_filename)
    {
        (r.clone(), f.clone())
    } else {
        match model_name {
            "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => (
                known_models::QWEN3_0_6B_GGUF_REPO.into(),
                known_models::QWEN3_0_6B_GGUF_FILE.into(),
            ),
            "gpt2" | "gpt2-q8_0" => (
                known_models::GPT2_GGUF_REPO.into(),
                known_models::GPT2_GGUF_FILE.into(),
            ),
            _ => return Err(Error::UnsupportedModel(model_name.to_string())),
        }
    };

    let gguf_path = config.core.base_models_dir().join(&filename);

    if !gguf_path.exists() {
        return Err(Error::Cache(format!(
            "GGUF model not cached at {}. Use hf-hub to download first.",
            gguf_path.display()
        )));
    }

    let lower = model_name.to_lowercase();
    if lower.contains("qwen") {
        Ok(Box::new(
            Qwen3Model::from_gguf(&gguf_path, device)
                .map_err(|e| Error::Other(format!("Qwen3 load error: {e}")))?,
        ))
    } else if lower.contains("gpt2") {
        Ok(Box::new(Gpt2Model::from_gguf(&gguf_path, device).map_err(
            |e| Error::Other(format!("GPT-2 load error: {e}")),
        )?))
    } else {
        Err(Error::UnsupportedModel(model_name.to_string()))
    }
}
