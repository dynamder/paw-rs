use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::PawConfig;
use crate::error::{Error, Result};
use crate::types::{BaseModelInfo, BundleMeta, LocalSdkInfo, RuntimeManifest};

/// Manages local caching of base models, programs, runtimes, and slug mappings.
///
/// Cache structure:
///   ~/.cache/programasweights/
///       base_models/
///           Qwen3-0.6B/
///           gpt2/
///       programs/
///           <program_id>/
///               adapter.gguf
///               prompt_template.txt
///               meta.json
///       runtimes/
///           <runtime_id>.json
///       slug_cache.json
#[derive(Debug, Clone)]
pub struct CacheManager {
    cache_dir: PathBuf,
}

/// Pre-configured base model repositories known to work with PAW.
pub mod known_models {
    pub const QWEN3_0_6B: &str = "Qwen/Qwen3-0.6B";
    pub const QWEN3_0_6B_GGUF_REPO: &str = "programasweights/Qwen3-0.6B-GGUF-Q6_K";
    pub const QWEN3_0_6B_GGUF_FILE: &str = "qwen3-0.6b-q6_k.gguf";

    pub const GPT2: &str = "openai-community/gpt2";
    pub const GPT2_GGUF_REPO: &str = "programasweights/GPT2-GGUF-Q8_0";
    pub const GPT2_GGUF_FILE: &str = "gpt2-q8_0.gguf";

    pub fn interpreter_to_gguf(interpreter: &str) -> Option<(&'static str, &'static str)> {
        match interpreter {
            "Qwen/Qwen3-0.6B" => Some((QWEN3_0_6B_GGUF_REPO, QWEN3_0_6B_GGUF_FILE)),
            "gpt2" => Some((GPT2_GGUF_REPO, GPT2_GGUF_FILE)),
            _ => None,
        }
    }
}

fn legacy_runtime_manifests() -> HashMap<&'static str, RuntimeManifest> {
    let mut m = HashMap::new();

    m.insert(
        "qwen3-0.6b-q6_k",
        RuntimeManifest {
            runtime_id: "qwen3-0.6b-q6_k".into(),
            manifest_version: 1,
            display_name: "Qwen3 0.6B (Q6_K)".into(),
            interpreter: known_models::QWEN3_0_6B.into(),
            adapter_format: "gguf_lora".into(),
            local_sdk: Some(LocalSdkInfo {
                supported: true,
                base_model: Some(BaseModelInfo {
                    provider: "huggingface".into(),
                    repo: known_models::QWEN3_0_6B_GGUF_REPO.into(),
                    filename: known_models::QWEN3_0_6B_GGUF_FILE.into(),
                    url: Some(format!(
                        "https://huggingface.co/{}/resolve/main/{}",
                        known_models::QWEN3_0_6B_GGUF_REPO,
                        known_models::QWEN3_0_6B_GGUF_FILE
                    )),
                    sha256: None,
                }),
                n_ctx: 2048,
            }),
        },
    );

    m.insert(
        "gpt2-q8_0",
        RuntimeManifest {
            runtime_id: "gpt2-q8_0".into(),
            manifest_version: 1,
            display_name: "GPT-2 124M (Q8_0)".into(),
            interpreter: known_models::GPT2.into(),
            adapter_format: "gguf_lora".into(),
            local_sdk: Some(LocalSdkInfo {
                supported: true,
                base_model: Some(BaseModelInfo {
                    provider: "huggingface".into(),
                    repo: known_models::GPT2_GGUF_REPO.into(),
                    filename: known_models::GPT2_GGUF_FILE.into(),
                    url: Some(format!(
                        "https://huggingface.co/{}/resolve/main/{}",
                        known_models::GPT2_GGUF_REPO,
                        known_models::GPT2_GGUF_FILE
                    )),
                    sha256: None,
                }),
                n_ctx: 2048,
            }),
        },
    );

    m
}

/// Look up a legacy runtime manifest by runtime_id.
pub fn legacy_runtime_manifest(runtime_id: &str) -> Option<RuntimeManifest> {
    let manifests = legacy_runtime_manifests();
    manifests.get(runtime_id).cloned()
}

impl CacheManager {
    /// Create a cache manager using the provided configuration.
    pub fn new(config: &PawConfig) -> Self {
        let cache_dir = config.cache_dir().clone();
        std::fs::create_dir_all(cache_dir.join("base_models")).ok();
        std::fs::create_dir_all(cache_dir.join("programs")).ok();
        std::fs::create_dir_all(cache_dir.join("runtimes")).ok();
        Self { cache_dir }
    }

    /// Create a cache manager with an explicit cache directory.
    pub fn new_with_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    // ── Base Models ─────────────────────────────────────────────────────

    pub fn base_models_dir(&self) -> PathBuf {
        self.cache_dir.join("base_models")
    }

    pub fn base_model_path(&self, repo: &str) -> PathBuf {
        let name = repo.replace('/', "--");
        self.base_models_dir().join(name)
    }

    pub fn is_base_model_cached(&self, repo: &str) -> bool {
        let dir = self.base_model_path(repo);
        if !dir.is_dir() {
            return false;
        }
        dir.join("config.json").exists()
            && (dir.join("model.safetensors.index.json").exists()
                || std::fs::read_dir(&dir)
                    .ok()
                    .map(|mut entries| {
                        entries.any(|e| {
                            e.ok()
                                .and_then(|e| e.path().extension().map(|ext| ext == "safetensors"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false))
    }

    pub fn is_gguf_cached(&self, filename: &str) -> bool {
        self.base_models_dir().join(filename).exists()
    }

    pub fn gguf_path(&self, filename: &str) -> PathBuf {
        self.base_models_dir().join(filename)
    }

    pub fn store_gguf(&self, filename: &str, bytes: &[u8]) -> Result<PathBuf> {
        let path = self.gguf_path(filename);
        std::fs::write(&path, bytes).map_err(Error::Io)?;
        Ok(path)
    }

    // ── Programs ────────────────────────────────────────────────────────

    /// Returns the cached program directory if the program is fully cached.
    pub fn get(&self, program_id: &str) -> Option<PathBuf> {
        let dir = self.program_dir(program_id);
        if dir.join("adapter.gguf").exists()
            && dir.join("prompt_template.txt").exists()
            && dir.join("meta.json").exists()
        {
            Some(dir)
        } else {
            None
        }
    }

    /// Returns the path for a program, whether cached or not.
    pub fn path_for(&self, program_id: &str) -> PathBuf {
        self.program_dir(program_id)
    }

    /// Store a downloaded .paw ZIP, extract it, and return the program directory.
    pub fn store(&self, program_id: &str, zip_bytes: &[u8]) -> Result<PathBuf> {
        let program_dir = self.program_dir(program_id);
        std::fs::create_dir_all(&program_dir)?;

        let paw_path = program_dir.join(format!("{program_id}.paw"));
        std::fs::write(&paw_path, zip_bytes)?;

        let reader = std::io::Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(reader)?;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();

            if name.contains("..") || name.starts_with('/') {
                return Err(Error::UnsafePath(name.into()));
            }

            let out_path = program_dir.join(&name);
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            std::fs::write(&out_path, data)?;
        }

        if self.get(program_id).is_none() {
            return Err(Error::MissingFile(format!(
                "Program {program_id} missing required files after extraction"
            )));
        }

        Ok(program_dir)
    }

    /// Write a resolved runtime manifest into the program's `meta.json`.
    ///
    /// Pure I/O — does no network calls. Caller is responsible for
    /// obtaining the `RuntimeManifest` (from cache, server, or legacy).
    pub fn hydrate_manifest(&self, program_dir: &Path, manifest: &RuntimeManifest) -> Result<()> {
        let meta_path = program_dir.join("meta.json");
        let meta_content = std::fs::read_to_string(&meta_path)?;
        let mut meta: BundleMeta = serde_json::from_str(&meta_content)?;

        meta.runtime = Some(manifest.clone());
        meta.runtime_manifest_version = Some(manifest.manifest_version);

        let content = serde_json::to_string_pretty(&meta)?;
        std::fs::write(&meta_path, content)?;
        Ok(())
    }

    fn program_dir(&self, program_id: &str) -> PathBuf {
        self.cache_dir.join("programs").join(program_id)
    }

    // ── Runtimes ────────────────────────────────────────────────────────

    pub fn get_cached_runtime_manifest(&self, runtime_id: &str) -> Option<RuntimeManifest> {
        if let Some(manifest) = legacy_runtime_manifest(runtime_id) {
            return Some(manifest);
        }

        let path = self.runtimes_dir().join(format!("{runtime_id}.json"));
        if !path.exists() {
            return None;
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).ok(),
            Err(_) => None,
        }
    }

    pub fn save_runtime_manifest(&self, manifest: &RuntimeManifest) {
        let path = self
            .runtimes_dir()
            .join(format!("{}.json", manifest.runtime_id));
        if let Ok(content) = serde_json::to_string_pretty(manifest) {
            let _ = std::fs::write(path, content);
        }
    }

    fn runtimes_dir(&self) -> PathBuf {
        self.cache_dir.join("runtimes")
    }

    // ── Slug Cache ──────────────────────────────────────────────────────

    pub fn get_cached_slug(&self, slug: &str) -> Option<String> {
        let cache = self.load_slug_cache();
        cache.get(slug).and_then(|id| {
            if self.get(id).is_some() {
                Some(id.clone())
            } else {
                None
            }
        })
    }

    pub fn save_slug_mapping(&self, slug: &str, program_id: &str) {
        let mut cache = self.load_slug_cache();
        cache.insert(slug.to_string(), program_id.to_string());
        let path = self.cache_dir.join("slug_cache.json");
        if let Ok(content) = serde_json::to_string(&cache) {
            let _ = std::fs::write(path, content);
        }
    }

    fn load_slug_cache(&self) -> HashMap<String, String> {
        let path = self.cache_dir.join("slug_cache.json");
        if !path.exists() {
            return HashMap::new();
        }
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default()
    }

    // ── Cache management ───────────────────────────────────────────────

    /// Calculate total cached size in bytes.
    pub fn total_size(&self) -> std::io::Result<u64> {
        fn dir_size(path: &Path) -> std::io::Result<u64> {
            if !path.exists() {
                return Ok(0);
            }
            let mut total = 0u64;
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let meta = entry.metadata()?;
                if meta.is_dir() {
                    total += dir_size(&entry.path())?;
                } else {
                    total += meta.len();
                }
            }
            Ok(total)
        }
        dir_size(&self.cache_dir)
    }

    /// Delete the entire cache.
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir).map_err(Error::Io)?;
        }
        Ok(())
    }

    /// Delete all downloaded programs, keeping base models.
    pub fn clear_programs(&self) -> Result<()> {
        let dir = self.cache_dir.join("programs");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(Error::Io)?;
        }
        Ok(())
    }

    /// Delete all base models, keeping downloaded programs.
    pub fn clear_base_models(&self) -> Result<()> {
        let dir = self.cache_dir.join("base_models");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(Error::Io)?;
        }
        Ok(())
    }
}
