use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{Error, Result};
use crate::format::tensor::TensorData;

// ── Constants ──────────────────────────────────────────────────────────

const PAW_VERSION: u32 = 2;
const DEFAULT_KIND: &str = "neural_program";
const DEFAULT_PREFIX_TYPE: &str = "kv_cache";
const DEFAULT_SOURCE: &str = "compiled";
const DEFAULT_LORA_ALPHA: f32 = 16.0;
const DEFAULT_MAX_NEW_TOKENS: u32 = 512;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_TOP_K: u32 = 50;

// ── PawFileMeta ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PawFileMeta {
    pub format_version: u32,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub interpreter_model: String,
    #[serde(default)]
    pub spec: String,
    #[serde(default)]
    pub pseudo_program: String,
    #[serde(default = "default_prefix_type")]
    pub prefix_type: String,
    #[serde(default)]
    pub prefix_steps: u32,
    #[serde(default)]
    pub num_layers: u32,
    #[serde(default)]
    pub has_lora: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub examples: Vec<ExamplePair>,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub source_info: HashMap<String, JsonValue>,
    #[serde(default)]
    pub prompt_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub lora_config: Option<LoRAConfig>,
    #[serde(default)]
    pub generation_config: Option<GenerationConfig>,
    #[serde(default)]
    pub base_model: String,
    #[serde(flatten)]
    pub extra: HashMap<String, JsonValue>,
}

fn default_kind() -> String {
    DEFAULT_KIND.to_string()
}
fn default_prefix_type() -> String {
    DEFAULT_PREFIX_TYPE.to_string()
}
fn default_source() -> String {
    DEFAULT_SOURCE.to_string()
}

impl Default for PawFileMeta {
    fn default() -> Self {
        Self {
            format_version: PAW_VERSION,
            kind: DEFAULT_KIND.into(),
            interpreter_model: String::new(),
            spec: String::new(),
            pseudo_program: String::new(),
            prefix_type: DEFAULT_PREFIX_TYPE.into(),
            prefix_steps: 0,
            num_layers: 0,
            has_lora: false,
            description: String::new(),
            author: String::new(),
            tags: vec![],
            examples: vec![],
            source: DEFAULT_SOURCE.into(),
            source_info: HashMap::new(),
            prompt_token_ids: None,
            lora_config: None,
            generation_config: Some(GenerationConfig::default()),
            base_model: String::new(),
            extra: HashMap::new(),
        }
    }
}

impl PawFileMeta {
    pub fn builder() -> PawFileMetaBuilder {
        PawFileMetaBuilder::default()
    }

    /// Validate the metadata and tensors together.
    pub fn validate(&self, tensors: &HashMap<String, TensorData>) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();

        if self.format_version != PAW_VERSION {
            errors.push(format!("Version {} != {PAW_VERSION}", self.format_version));
        }

        let model = if self.interpreter_model.is_empty() {
            &self.base_model
        } else {
            &self.interpreter_model
        };
        if model.is_empty() {
            errors.push("Missing interpreter_model".into());
        }

        if self.prefix_steps > 256 {
            errors.push(format!("prefix_steps {} > 256", self.prefix_steps));
        }

        if self.has_lora {
            if let Some(ref lc) = self.lora_config {
                if lc.rank > 128 {
                    errors.push(format!("LoRA rank {} > 128", lc.rank));
                }
            }
            let n = tensors.keys().filter(|k| k.starts_with("lora_")).count();
            if n == 0 {
                errors.push("has_lora=True but no LoRA tensors".into());
            }
        }

        let actual = tensors.keys().filter(|k| k.ends_with("_key")).count();
        if self.num_layers as usize != actual {
            errors.push(format!(
                "{} layers declared, {} found",
                self.num_layers, actual
            ));
        }

        let meta_str = serde_json::to_string(self).unwrap_or_default();
        for pat in &["__import__", "eval(", "exec(", "os.system", "subprocess"] {
            if meta_str.contains(pat) {
                errors.push(format!("Suspicious: {pat}"));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Format(errors.join("; ")))
        }
    }
}

// ── Builder ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct PawFileMetaBuilder {
    interpreter_model: Option<String>,
    spec: Option<String>,
    pseudo_program: Option<String>,
    description: Option<String>,
    author: Option<String>,
    tags: Option<Vec<String>>,
    examples: Option<Vec<ExamplePair>>,
    source: Option<String>,
    source_info: Option<HashMap<String, JsonValue>>,
    prompt_token_ids: Option<Vec<u32>>,
    lora_config: Option<LoRAConfig>,
    generation_config: Option<GenerationConfig>,
    base_model: Option<String>,
}

impl PawFileMetaBuilder {
    pub fn interpreter_model(mut self, v: impl Into<String>) -> Self {
        self.interpreter_model = Some(v.into());
        self
    }
    pub fn spec(mut self, v: impl Into<String>) -> Self {
        self.spec = Some(v.into());
        self
    }
    pub fn pseudo_program(mut self, v: impl Into<String>) -> Self {
        self.pseudo_program = Some(v.into());
        self
    }
    pub fn description(mut self, v: impl Into<String>) -> Self {
        self.description = Some(v.into());
        self
    }
    pub fn author(mut self, v: impl Into<String>) -> Self {
        self.author = Some(v.into());
        self
    }
    pub fn tags(mut self, v: Vec<String>) -> Self {
        self.tags = Some(v);
        self
    }
    pub fn examples(mut self, v: Vec<ExamplePair>) -> Self {
        self.examples = Some(v);
        self
    }
    pub fn source(mut self, v: impl Into<String>) -> Self {
        self.source = Some(v.into());
        self
    }
    pub fn lora_config(mut self, v: LoRAConfig) -> Self {
        self.lora_config = Some(v);
        self
    }
    pub fn generation_config(mut self, v: GenerationConfig) -> Self {
        self.generation_config = Some(v);
        self
    }
    pub fn base_model(mut self, v: impl Into<String>) -> Self {
        self.base_model = Some(v.into());
        self
    }

    pub fn build(self) -> PawFileMeta {
        let d = PawFileMeta::default();
        PawFileMeta {
            interpreter_model: self.interpreter_model.unwrap_or(d.interpreter_model),
            spec: self.spec.unwrap_or(d.spec),
            pseudo_program: self.pseudo_program.unwrap_or(d.pseudo_program),
            description: self.description.unwrap_or(d.description),
            author: self.author.unwrap_or(d.author),
            tags: self.tags.unwrap_or(d.tags),
            examples: self.examples.unwrap_or(d.examples),
            source: self.source.unwrap_or(d.source),
            source_info: self.source_info.unwrap_or(d.source_info),
            prompt_token_ids: self.prompt_token_ids.or(d.prompt_token_ids),
            lora_config: self.lora_config.or(d.lora_config),
            generation_config: self.generation_config.or(d.generation_config),
            base_model: self.base_model.unwrap_or(d.base_model),
            ..d
        }
    }
}

// ── Supporting types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamplePair {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: u32,
    #[serde(default = "default_lora_alpha")]
    pub alpha: f32,
    #[serde(default)]
    pub target_modules: Vec<String>,
}

const fn default_lora_alpha() -> f32 {
    DEFAULT_LORA_ALPHA
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

const fn default_max_new_tokens() -> u32 {
    DEFAULT_MAX_NEW_TOKENS
}
const fn default_top_p() -> f32 {
    DEFAULT_TOP_P
}
const fn default_top_k() -> u32 {
    DEFAULT_TOP_K
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            temperature: 0.0,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
        }
    }
}
