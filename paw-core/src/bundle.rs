use std::io::Read;
use std::path::{Path, PathBuf};

use tracing::info;

use crate::cache::known_models;
use crate::error::{Error, Result};
use crate::types::BundleMeta;
use crate::types::RuntimeManifest;

/// Parsed contents of a .paw bundle directory.
///
/// A `.paw` program bundle is a ZIP archive extracted to
/// `~/.cache/programasweights/programs/<id>/` containing:
/// - `adapter.gguf` — Q4_0 GGUF LoRA adapter
/// - `prompt_template.txt` — pre-rendered chat template with `{INPUT_PLACEHOLDER}`
/// - `meta.json` — program metadata
#[derive(Debug, Clone)]
pub struct PawBundle {
    /// Root directory of the extracted bundle.
    pub program_dir: PathBuf,
    /// Parsed metadata.
    pub meta: BundleMeta,
    /// Pre-rendered prompt template text.
    pub prompt_template: String,
    /// Path to the GGUF LoRA adapter file.
    pub adapter_path: PathBuf,
}

impl PawBundle {
    /// Load a bundle from an already-extracted program directory.
    pub fn load_from_dir<P: AsRef<Path>>(program_dir: P) -> Result<Self> {
        let dir = program_dir.as_ref().to_path_buf();

        let meta_path = dir.join("meta.json");
        let meta_content = std::fs::read_to_string(&meta_path)
            .map_err(|e| Error::MissingFile(format!("meta.json: {e}")))?;
        let mut meta: BundleMeta = serde_json::from_str(&meta_content)?;

        if meta.interpreter.is_empty() {
            meta.interpreter = infer_interpreter(&meta);
        }

        let template_path = dir.join("prompt_template.txt");
        let prompt_template = std::fs::read_to_string(&template_path)
            .map_err(|e| Error::MissingFile(format!("prompt_template.txt: {e}")))?;

        let adapter_path = dir.join("adapter.gguf");
        if !adapter_path.exists() {
            return Err(Error::MissingFile(format!("adapter.gguf not in {dir:?}")));
        }

        info!("Loaded program bundle from {}", dir.display());
        Ok(Self {
            program_dir: dir,
            meta,
            prompt_template,
            adapter_path,
        })
    }

    /// Parse a .paw ZIP archive from raw bytes and extract it to a directory.
    pub fn from_zip<P: AsRef<Path>>(zip_bytes: &[u8], extract_dir: P) -> Result<Self> {
        let dir = extract_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let reader = std::io::Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(reader)?;

        for i in 0..archive.len() {
            let entry = archive.by_index(i)?;
            let name = entry.name().to_string();
            if name.contains("..") || name.starts_with('/') {
                return Err(Error::UnsafePath(name.into()));
            }
        }

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();
            let out_path = dir.join(&name);
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            std::fs::write(&out_path, data)?;
        }

        let bundle = Self::load_from_dir(&dir)?;
        info!("Extracted .paw bundle to {}", dir.display());
        Ok(bundle)
    }

    /// Split the prompt template at `{INPUT_PLACEHOLDER}`.
    ///
    /// Returns `(prefix_text, suffix_text)` where:
    /// - `prefix_text` is everything before the placeholder
    /// - `suffix_text` is everything after the placeholder
    pub fn split_template(&self) -> (String, String) {
        let placeholder = "{INPUT_PLACEHOLDER}";
        if let Some(pos) = self.prompt_template.find(placeholder) {
            let prefix = self.prompt_template[..pos].to_string();
            let suffix = self.prompt_template[pos + placeholder.len()..].to_string();
            (prefix, suffix)
        } else {
            (self.prompt_template.clone(), String::new())
        }
    }

    /// Get the interpreter model identifier from the bundle metadata.
    pub fn interpreter_model(&self) -> &str {
        if !self.meta.interpreter.is_empty() {
            &self.meta.interpreter
        } else {
            "Qwen/Qwen3-0.6B"
        }
    }
}

impl BundleMeta {
    /// Resolve runtime manifest from embedded data, cache, or legacy fallback.
    pub fn resolve_runtime_manifest(&self) -> Option<RuntimeManifest> {
        if let Some(ref runtime) = self.runtime {
            let has_complete_info = runtime
                .local_sdk
                .as_ref()
                .and_then(|s| s.base_model.as_ref())
                .is_some();
            if has_complete_info {
                return Some(runtime.clone());
            }
        }

        let runtime_id = self.runtime_id.as_deref()?;
        crate::cache::legacy_runtime_manifest(runtime_id)
    }
}

fn infer_interpreter(meta: &BundleMeta) -> String {
    if let Some(ref runtime) = meta.runtime {
        if !runtime.interpreter.is_empty() {
            return runtime.interpreter.clone();
        }
    }

    if let Some(ref rid) = meta.runtime_id {
        return match rid.as_str() {
            "qwen3-0.6b-q6_k" => known_models::QWEN3_0_6B.to_string(),
            "gpt2-q8_0" => known_models::GPT2.to_string(),
            _ => rid.clone(),
        };
    }

    if let Some(v) = meta.extra.get("base_model").and_then(|v| v.as_str()) {
        return v.to_string();
    }

    "Qwen/Qwen3-0.6B".to_string()
}
