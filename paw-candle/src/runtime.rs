use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use paw_core::{Error, PawBundle, PawFnTrait, PawRuntimeOptions};

use crate::config::{DevicePreference, PawCandleConfig};
use crate::kv_cache::PrefixKvCache;
use crate::lora::GgufLoraAdapter;
use crate::models::{QuantizedModel, gpt2::Gpt2Model, qwen3::Qwen3Model};
use crate::pool::{self, ModelPool};
use crate::tokenizer::Tokenizer;

// ── PawFunction (inference runtime) ────────────────────────────────────

/// The inference runtime for a PAW program.
///
/// Created by [`PawFnLoader`] — this struct only owns loaded state and
/// exposes [`run()`](PawFunction::run).
pub struct PawFunction {
    pool: Arc<ModelPool>,
    model: Arc<Mutex<Box<dyn QuantizedModel>>>,
    tokenizer: Tokenizer,
    prefix_text: String,
    suffix_text: String,
    n_prefix: usize,
    n_ctx: usize,
    #[allow(dead_code)]
    lora_adapter: Option<GgufLoraAdapter>,
    kv_cache: PrefixKvCache,
    prefix_loaded: bool,
    eos_token_id: u32,
    interpreter_model: String,
}

fn sample_logits(logits: &Tensor, opts: &PawRuntimeOptions) -> Result<u32, Error> {
    if opts.temperature <= 0.0 {
        let id = logits
            .argmax(0)
            .map_err(|e| Error::Other(format!("argmax: {e}")))?
            .to_scalar::<u32>()
            .map_err(|e| Error::Other(format!("to_scalar: {e}")))?;
        return Ok(id);
    }

    let temperature = opts.temperature.max(1e-8) as f64;
    let scaled = (logits / temperature).map_err(|e| Error::Other(format!("temperature: {e}")))?;

    let probs = softmax(&scaled, 0).map_err(|e| Error::Other(format!("softmax: {e}")))?;

    let shape = probs.shape().clone();
    let device = probs.device();
    let uniform = Tensor::rand(1e-8f64, 1.0 - 1e-8, &shape, device)
        .map_err(|e| Error::Other(format!("rand: {e}")))?;

    let gumbel = uniform
        .log()
        .map_err(|e| Error::Other(format!("log: {e}")))?
        .neg()
        .map_err(|e| Error::Other(format!("neg: {e}")))?
        .log()
        .map_err(|e| Error::Other(format!("log2: {e}")))?
        .neg()
        .map_err(|e| Error::Other(format!("neg2: {e}")))?;

    let perturbed = (scaled + gumbel).map_err(|e| Error::Other(format!("perturb: {e}")))?;
    let id = perturbed
        .argmax(0)
        .map_err(|e| Error::Other(format!("sample: {e}")))?
        .to_scalar::<u32>()
        .map_err(|e| Error::Other(format!("sample scalar: {e}")))?;
    Ok(id)
}

impl PawFunction {
    pub(crate) fn new(
        pool: Arc<ModelPool>,
        model: Arc<Mutex<Box<dyn QuantizedModel>>>,
        tokenizer: Tokenizer,
        prefix_text: String,
        suffix_text: String,
        n_prefix: usize,
        n_ctx: usize,
        #[allow(dead_code)] lora_adapter: Option<GgufLoraAdapter>,
        kv_cache: PrefixKvCache,
        prefix_loaded: bool,
        eos_token_id: u32,
        interpreter_model: String,
    ) -> Self {
        Self {
            pool,
            model,
            tokenizer,
            prefix_text,
            suffix_text,
            n_prefix,
            n_ctx,
            lora_adapter,
            kv_cache,
            prefix_loaded,
            eos_token_id,
            interpreter_model,
        }
    }

    pub fn interpreter(&self) -> &str {
        &self.interpreter_model
    }

    /// Run inference on the given input text.
    pub fn run(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        let _permit = self.pool.acquire()?;
        let full_input = format!("{}{}", input, self.suffix_text);

        let mut model = self.model.lock().unwrap();

        // Swap LoRA to this program's adapter
        if let Some(ref lora) = self.lora_adapter {
            model.set_lora(lora);
        }

        if self.prefix_loaded {
            let input_tokens = self.tokenizer.encode(&full_input)?;

            if let Some(ref prefix_kv) = self.kv_cache.get_cached() {
                model.set_prefix_cache(prefix_kv);
            }

            if input_tokens.len() >= self.n_ctx {
                return Err(Error::Other(format!(
                    "Input too long: {}",
                    input_tokens.len()
                )));
            }

            let gen_limit = opts
                .max_tokens
                .map(|m| m.min(self.n_ctx - input_tokens.len()))
                .unwrap_or(self.n_ctx - input_tokens.len());

            let device = model.device().clone();
            let te = |e: candle_core::Error| Error::Other(format!("tensor op: {e}"));

            let prefill_tensor = Tensor::new(&input_tokens[..], &device)
                .map_err(&te)?
                .unsqueeze(0)
                .map_err(&te)?;
            let logits = model
                .forward(&prefill_tensor, self.n_prefix)
                .map_err(|e| Error::Other(format!("prefill: {e}")))?;
            let last_logits = logits
                .squeeze(0)
                .map_err(&te)?
                .get(logits.dim(1).map_err(&te)? - 1)
                .map_err(&te)?;
            let mut next_id = sample_logits(&last_logits, opts)?;

            let mut all_ids = input_tokens.clone();
            if next_id != self.eos_token_id {
                all_ids.push(next_id);
            }

            let start_pos = self.n_prefix + input_tokens.len();
            let mut current_pos = start_pos;
            for step in 0..gen_limit {
                if next_id == self.eos_token_id {
                    break;
                }
                if step > 0 {
                    let inp = Tensor::new(&[next_id], &device)
                        .map_err(&te)?
                        .unsqueeze(0)
                        .map_err(&te)?;
                    let logits = model
                        .forward(&inp, current_pos)
                        .map_err(|e| Error::Other(format!("decode: {e}")))?;
                    let last = logits.squeeze(0).map_err(&te)?.get(0).map_err(&te)?;
                    next_id = sample_logits(&last, opts)?;
                    if next_id == self.eos_token_id {
                        break;
                    }
                    all_ids.push(next_id);
                }
                current_pos += 1;
            }

            let output = self.tokenizer.decode(&all_ids[input_tokens.len()..])?;
            return Ok(output);
        }

        // FIRST RUN: tokenize prefix + input + suffix together
        let full_prompt = format!("{}{}", self.prefix_text, &full_input);
        let all_tokens = self.tokenizer.encode(&full_prompt)?;

        if all_tokens.len() >= self.n_ctx {
            return Err(Error::Other(format!(
                "Input too long: {} tokens",
                all_tokens.len()
            )));
        }

        let gen_limit = opts
            .max_tokens
            .map(|m| m.min(self.n_ctx - all_tokens.len()))
            .unwrap_or(self.n_ctx - all_tokens.len());

        let device = model.device().clone();
        let te = |e: candle_core::Error| Error::Other(format!("tensor op: {e}"));

        // Prefill
        let prefill_tensor = Tensor::new(&all_tokens[..], &device)
            .map_err(&te)?
            .unsqueeze(0)
            .map_err(&te)?;
        let logits = model
            .forward(&prefill_tensor, 0)
            .map_err(|e| Error::Other(format!("prefill: {e}")))?;
        let last_logits = logits
            .squeeze(0)
            .map_err(&te)?
            .get(logits.dim(1).map_err(&te)? - 1)
            .map_err(&te)?;
        let mut next_id = sample_logits(&last_logits, opts)?;

        let mut gen_ids = Vec::new();
        if next_id != self.eos_token_id {
            gen_ids.push(next_id);
        }

        let mut current_pos = all_tokens.len();
        for step in 0..gen_limit {
            if next_id == self.eos_token_id {
                break;
            }
            if step > 0 {
                let inp = Tensor::new(&[next_id], &device)
                    .map_err(&te)?
                    .unsqueeze(0)
                    .map_err(&te)?;
                let logits = model
                    .forward(&inp, current_pos)
                    .map_err(|e| Error::Other(format!("decode: {e}")))?;
                let last = logits.squeeze(0).map_err(&te)?.get(0).map_err(&te)?;
                next_id = sample_logits(&last, opts)?;
                if next_id == self.eos_token_id {
                    break;
                }
                gen_ids.push(next_id);
            }
            current_pos += 1;
        }

        // Save prefix KV cache for future runs
        let input_token_len = self.tokenizer.encode(&full_input)?.len();
        let n_prefix_actual = all_tokens.len() - input_token_len;
        if let Some(prefix_kv) = model.extract_prefix_cache(n_prefix_actual) {
            self.kv_cache.set_cache(prefix_kv.clone());
            self.kv_cache.save(&prefix_kv).ok();
            model.set_prefix_cache(&prefix_kv);
        }
        self.n_prefix = n_prefix_actual;
        self.prefix_loaded = true;

        let output = self.tokenizer.decode(&gen_ids)?;
        Ok(output)
    }
}

impl PawFnTrait for PawFunction {
    fn run(&mut self, input: &str) -> Result<String, Error> {
        self.run(input, &PawRuntimeOptions::default())
    }
    fn run_with(&mut self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        self.run(input, opts)
    }
    fn interpreter(&self) -> &str {
        self.interpreter()
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
        let model_name = bundle.interpreter_model().to_string();
        let max_copies = self.config.max_model_copies;
        let config = self.config.clone();
        let gguf_filename = resolve_gguf(&bundle, &config);
        let base_models_dir = config.core.base_models_dir();
        let gguf_path = base_models_dir.join(&gguf_filename);

        let (model, pool) = pool::get_or_load_model(&model_name, max_copies, {
            let mn = model_name.clone();
            let gp = gguf_path.clone();
            let dev = device.clone();
            move || load_model_standalone(&mn, &gp, &dev)
        })?;

        let tokenizer = Tokenizer::new(&bundle)?;
        self.assemble(bundle, pool, model, tokenizer)
    }

    /// Load with a pre-existing (possibly shared) model.
    pub fn load_with_model(
        self,
        pool: Arc<ModelPool>,
        model: Arc<Mutex<Box<dyn QuantizedModel>>>,
    ) -> Result<PawFunction, Error> {
        let bundle = self.load_bundle()?;
        let tokenizer = Tokenizer::new(&bundle)?;
        self.assemble(bundle, pool, model, tokenizer)
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
        pool: Arc<ModelPool>,
        model: Arc<Mutex<Box<dyn QuantizedModel>>>,
        tokenizer: Tokenizer,
    ) -> Result<PawFunction, Error> {
        let device;
        {
            let m = model.lock().unwrap();
            device = m.device().clone();
        }
        let (num_layers, head_dim, num_kv_heads) = {
            let m = model.lock().unwrap();
            (m.num_layers(), m.head_dim(), m.num_kv_heads())
        };
        let lora = GgufLoraAdapter::from_gguf_file(&bundle.adapter_path, &device)
            .map_err(|e| Error::Other(format!("LoRA adapter load failed: {e}")))?;
        {
            let mut m = model.lock().unwrap();
            m.set_lora(&lora);
        }
        let lora = Some(lora);
        let (prefix_text, suffix_text) = bundle.split_template();

        let placeholder = "x";
        let full_test = format!("{prefix_text}{placeholder}{suffix_text}");
        let full_test_tokens = tokenizer.encode(&full_test)?;
        let placeholder_tokens = tokenizer.encode(placeholder)?;
        let n_prefix = full_test_tokens
            .len()
            .saturating_sub(placeholder_tokens.len());
        let n_ctx = self.config.core.n_ctx() as usize;

        let mut kv_cache = PrefixKvCache::new(
            bundle.program_dir.join("prefix_kv_cache.bin"),
            num_layers,
            head_dim,
            num_kv_heads,
            n_prefix,
            &device,
        );

        let prefix_loaded = kv_cache.try_load().unwrap_or(false);
        if prefix_loaded {
            if let Some(ref cached) = kv_cache.get_cached() {
                let mut m = model.lock().unwrap();
                m.set_prefix_cache(cached);
            }
        }

        let interpreter = bundle.interpreter_model().to_string();

        let eos_token_id = tokenizer.eos_token_id();

        Ok(PawFunction::new(
            pool,
            model,
            tokenizer,
            prefix_text,
            suffix_text,
            n_prefix,
            n_ctx,
            lora,
            kv_cache,
            prefix_loaded,
            eos_token_id,
            interpreter,
        ))
    }
}

// ── Unified asset downloader ──────────────────────────────────────────

/// Ensure the base model GGUF and tokenizer are cached locally.
///
/// Downloads from HuggingFace if not already present.  Both the CLI and
/// benchmark examples should call this function with the same arguments so
/// they always use identical assets.
///
/// # Arguments
/// * `config`     — user config (for `base_models_dir`)
/// * `program_dir` — downloaded program bundle directory
/// * `interpreter` — model identifier, e.g. `"Qwen/Qwen3-0.6B"`
/// Download the GGUF base model and tokenizer for the given interpreter.
///
/// This is the single unified path that both the CLI and benchmark
/// examples use — any consumer of paw-candle should call this before
/// [`PawFnLoader::load`].
pub async fn ensure_assets(
    config: &paw_core::PawConfig,
    program_dir: &Path,
    interpreter: &str,
) -> Result<(), paw_core::Error> {
    use paw_core::cache::known_models;

    let hf =
        hf_hub::HFClient::new().map_err(|e| paw_core::Error::Other(format!("hf-hub init: {e}")))?;

    let (repo, file, tok_owner, tok_model) = match interpreter {
        "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => (
            known_models::QWEN3_0_6B_GGUF_REPO,
            known_models::QWEN3_0_6B_GGUF_FILE,
            "Qwen",
            "Qwen3-0.6B",
        ),
        "gpt2" | "gpt2-q8_0" => (
            known_models::GPT2_GGUF_REPO,
            known_models::GPT2_GGUF_FILE,
            "openai-community",
            "gpt2",
        ),
        other => return Err(paw_core::Error::UnsupportedModel(other.to_string())),
    };

    // GGUF
    let gguf_path = config.base_models_dir().join(file);
    if !gguf_path.exists() {
        let cached = hf
            .model(repo, "")
            .download_file()
            .filename(file)
            .send()
            .await
            .map_err(|e| paw_core::Error::Other(format!("hf-hub GGUF: {e}")))?;
        if let Some(p) = gguf_path.parent() {
            std::fs::create_dir_all(p).map_err(paw_core::Error::Io)?;
        }
        std::fs::copy(&cached, &gguf_path).map_err(paw_core::Error::Io)?;
    }

    // Tokenizer
    let tok_path = program_dir.join("tokenizer.json");
    if !tok_path.exists() {
        let cached = hf
            .model(tok_owner, tok_model)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .await
            .map_err(|e| paw_core::Error::Other(format!("hf-hub tokenizer: {e}")))?;
        std::fs::copy(&cached, &tok_path).map_err(paw_core::Error::Io)?;
    }

    Ok(())
}

// ── Private helpers ─────────────────────────────────────────────────

fn select_device(config: &PawCandleConfig) -> Result<Device, Error> {
    match config.device {
        DevicePreference::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(d) = Device::new_cuda(0) {
                return Ok(d);
            }
            #[cfg(feature = "metal")]
            if let Ok(d) = Device::new_metal(0) {
                return Ok(d);
            }
            Ok(Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        DevicePreference::Cuda => Device::new_cuda(0).map_err(|e| Error::Other(e.to_string())),
        #[cfg(feature = "metal")]
        DevicePreference::Metal => Device::new_metal(0).map_err(|e| Error::Other(e.to_string())),
        DevicePreference::Cpu => Ok(Device::Cpu),
    }
}

/// Select the device for loading models, exposed for use by paw-rs.
pub fn select_device_for_loading(config: &PawCandleConfig) -> Result<Device, Error> {
    select_device(config)
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

fn resolve_gguf(bundle: &PawBundle, config: &PawCandleConfig) -> String {
    use paw_core::cache::known_models;
    if let (Some(_), Some(f)) = (&config.base_model_repo, &config.gguf_filename) {
        return f.clone();
    }
    match bundle.interpreter_model() {
        "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => known_models::QWEN3_0_6B_GGUF_FILE.into(),
        "gpt2" | "gpt2-q8_0" => known_models::GPT2_GGUF_FILE.into(),
        _ => String::new(),
    }
}

fn load_model_standalone(
    model_name: &str,
    gguf_path: &Path,
    device: &Device,
) -> Result<Box<dyn QuantizedModel>, Error> {
    if !gguf_path.exists() {
        return Err(Error::Cache(format!(
            "GGUF model not cached at {}",
            gguf_path.display()
        )));
    }
    let lower = model_name.to_lowercase();
    if lower.contains("qwen") {
        Ok(Box::new(
            Qwen3Model::from_gguf(gguf_path, device)
                .map_err(|e| Error::Other(format!("Qwen3 load error: {e}")))?,
        ))
    } else if lower.contains("gpt2") {
        Ok(Box::new(Gpt2Model::from_gguf(gguf_path, device).map_err(
            |e| Error::Other(format!("GPT-2 load error: {e}")),
        )?))
    } else {
        Err(Error::UnsupportedModel(model_name.to_string()))
    }
}
