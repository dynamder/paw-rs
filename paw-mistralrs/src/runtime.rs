use std::path::PathBuf;
use std::sync::Arc;

use mistralrs::{GgufModelBuilder, RequestBuilder, TextMessageRole, blocking::BlockingModel};
use paw_core::{Error, PawBundle};

use crate::config::PawMistralRsConfig;
use crate::converter::convert_adapter_gguf_to_safetensors;
use crate::kv_cache::PrefixCache;
use crate::tokenizer::Tokenizer;

/// Sampling and generation parameters for inference.
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

/// The inference runtime using mistral.rs.
pub struct PawFunction {
    model: BlockingModel,
    tokenizer: Tokenizer,
    prefix_text: String,
    suffix_text: String,
    n_ctx: usize,
    eos_token_id: u32,
    prefix_cache: Option<PrefixCache>,
}

impl PawFunction {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: BlockingModel,
        tokenizer: Tokenizer,
        prefix_text: String,
        suffix_text: String,
        n_ctx: usize,
        eos_token_id: u32,
        prefix_cache: Option<PrefixCache>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            prefix_text,
            suffix_text,
            n_ctx,
            eos_token_id,
            prefix_cache,
        }
    }

    /// Run inference on the given input text.
    pub fn run(&self, input: &str, opts: &PawRuntimeOptions) -> Result<String, Error> {
        let full_input = format!("{}{}", input, self.suffix_text);
        let full_prompt = format!("{}{}", self.prefix_text, &full_input);

        let all_tokens = self.tokenizer.encode(&full_prompt)?;

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

        let request = RequestBuilder::new()
            .add_message(TextMessageRole::User, &full_prompt)
            .set_sampler_max_len(gen_limit)
            .set_sampler_temperature(opts.temperature)
            .set_sampler_topp(opts.top_p);

        let response = self
            .model
            .send_chat_request(request)
            .map_err(|e| Error::Other(format!("mistralrs inference: {e}")))?;

        let output = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.clone())
            .unwrap_or_default();

        if let Some(ref cache) = self.prefix_cache {
            let input_tokens = self.tokenizer.encode(&full_input).ok();
            if let Some(inp_tok) = input_tokens {
                let prefix_token_len = all_tokens.len() - inp_tok.len();
                let prefix_tokens: Vec<u32> = all_tokens[..prefix_token_len].to_vec();
                cache.save(&prefix_tokens).ok();
            }
        }

        Ok(output)
    }
}

/// Local loader for PAW program bundles, using mistral.rs.
pub struct PawFnLoader {
    program_dir: PathBuf,
    config: PawMistralRsConfig,
}

impl PawFnLoader {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            program_dir: dir.into(),
            config: PawMistralRsConfig::default(),
        }
    }

    pub fn config(mut self, config: PawMistralRsConfig) -> Self {
        self.config = config;
        self
    }

    pub fn load(self) -> Result<PawFunction, Error> {
        let bundle = self.load_bundle()?;
        let model = self.load_model(&bundle)?;
        let tokenizer = Tokenizer::new(&bundle)?;
        self.assemble(bundle, model, tokenizer)
    }

    pub fn load_bundle(&self) -> Result<PawBundle, Error> {
        PawBundle::load_from_dir(&self.program_dir)
    }

    pub fn load_model(&self, bundle: &PawBundle) -> Result<BlockingModel, Error> {
        use paw_core::cache::known_models;

        let model_name = bundle.interpreter_model();

        let filename = if let Some(ref f) = self.config.gguf_filename {
            f.clone()
        } else {
            match model_name {
                "Qwen/Qwen3-0.6B" | "qwen3-0.6b-q6_k" => known_models::QWEN3_0_6B_GGUF_FILE.into(),
                "gpt2" | "gpt2-q8_0" => known_models::GPT2_GGUF_FILE.into(),
                _ => return Err(Error::UnsupportedModel(model_name.to_string())),
            }
        };

        let gguf_path = self.config.core.base_models_dir().join(&filename);

        if !gguf_path.exists() {
            return Err(Error::Cache(format!(
                "GGUF model not cached at {}. Use hf-hub to download first.",
                gguf_path.display()
            )));
        }

        let gguf_dir = gguf_path
            .parent()
            .ok_or_else(|| Error::Other("Invalid GGUF path".into()))?
            .to_path_buf();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Other(format!("tokio runtime: {e}")))?;

        let async_model = rt
            .block_on(
                GgufModelBuilder::new(
                    gguf_dir
                        .to_str()
                        .ok_or_else(|| Error::Other("Invalid GGUF dir path".into()))?,
                    vec![filename],
                )
                .build(),
            )
            .map_err(|e| Error::Other(format!("mistralrs build: {e}")))?;

        let model = BlockingModel::new(async_model, Arc::new(rt));
        Ok(model)
    }

    pub fn load_model_with_lora(&self, bundle: &PawBundle) -> Result<BlockingModel, Error> {
        if !bundle.adapter_path.exists() {
            return self.load_model(bundle);
        }

        let lora_dir = bundle.program_dir.join("lora_safetensors");
        convert_adapter_gguf_to_safetensors(&bundle.adapter_path, &lora_dir)?;

        self.load_model(bundle)
    }

    pub fn assemble(
        &self,
        bundle: PawBundle,
        model: BlockingModel,
        tokenizer: Tokenizer,
    ) -> Result<PawFunction, Error> {
        let (prefix_text, suffix_text) = bundle.split_template();
        let n_prefix = tokenizer.encode(&prefix_text)?.len();
        let n_ctx = self.config.core.n_ctx() as usize;
        let eos_token_id = tokenizer.eos_token_id();

        let mut prefix_cache =
            PrefixCache::new(bundle.program_dir.join("prefix_cache.bin"), &prefix_text);
        let cache_loaded = prefix_cache.try_load().unwrap_or(false);

        Ok(PawFunction::new(
            model,
            tokenizer,
            prefix_text,
            suffix_text,
            n_ctx,
            eos_token_id,
            Some(prefix_cache),
        ))
    }
}
