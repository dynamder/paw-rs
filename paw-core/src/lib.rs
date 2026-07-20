//! # PAW Core — ProgramAsWeights Rust SDK core
//!
//! This crate provides the core types, HTTP client, cache management, and
//! bundle parsing for the ProgramAsWeights (PAW) ecosystem.
//!
//! It is the foundation of [`paw-rs`](https://crates.io/crates/paw-rs)
//! and [`paw-candle`](https://crates.io/crates/paw-candle).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use paw_core::prelude::*;
//!
//! # async fn example() -> Result<()> {
//! let config = PawConfig::from_env();
//! let client = PawClient::new(&config);
//!
//! // Compile a spec (calls remote API)
//! let program = client.compile(
//!     CompileRequest::builder().spec("Classify sentiment").build()?
//! ).await?;
//! println!("Program ID: {}", program.id);
//!
//! // Download the .paw bundle
//! let dir = client.download_paw(&program.id).await?;
//!
//! // Load and inspect the bundle
//! let bundle = PawBundle::load_from_dir(&dir)?;
//! println!("Spec: {}", bundle.meta.spec);
//! # Ok(())
//! # }
//! ```
//!
//! ## Configuration
//!
//! `PawConfig` reads from environment variables and supports a builder:
//!
//! ```rust,no_run
//! use paw_core::{PawConfig, PawClient, CompileRequest, Result};
//!
//! # async fn example() -> Result<()> {
//! let config = PawConfig::builder()
//!     .api_url("https://custom.example.com")
//!     .api_key("paw_sk_xxx")
//!     .n_ctx(4096)
//!     .verbose(true)
//!     .build()?;
//!
//! let client = PawClient::new(&config);
//! // Use client...
//! # Ok(())
//! # }
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Description | Default |
//! |---|---|---|
//! | `PAW_API_URL` | API server URL | `https://programasweights.com` |
//! | `PAW_API_KEY` | API key | (none) |
//! | `PAW_CACHE_DIR` | Cache root | `~/.cache/programasweights/` |
//! | `PAW_N_CTX` | Context window | `2048` |
//! | `PAW_GPU_LAYERS` | GPU layers | `-1` (all) |
//! | `PAW_VERBOSE` | Verbose logging | `false` |
//! | `PAW_OFFLINE` | Offline-only | `false` |
//!
//! ## Examples
//!
//! Runnable examples in [`paw-core/examples/`](https://github.com/dynamder/paw-rs/tree/main/paw-core/examples):
//!
//! | File | Description |
//! |------|-------------|
//! | [`download_and_save.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-core/examples/download_and_save.rs) | Download bundle, read format, binary roundtrip |
//! | [`clear_cache.rs`](https://github.com/dynamder/paw-rs/tree/main/paw-core/examples/clear_cache.rs) | Inspect and clear local cache |
//!
//! ## High-level convenience functions
//!
//! ```rust,no_run
//! use paw_core::{compile, resolve, login, clear_cache, PawConfig, Result};
//!
//! # async fn example() -> Result<()> {
//! let config = PawConfig::from_env();
//!
//! // One-shot compile (same as build + client.compile)
//! let program = compile("Classify sentiment", &config, None, None).await?;
//!
//! // Resolve a slug to a program ID
//! let id = resolve("email-triage", &config, false).await?;
//!
//! // Save an API key
//! login("paw_sk_xxx")?;
//!
//! // Clear the local cache
//! clear_cache(&config)?;
//! # Ok(())
//! # }
//! ```

pub mod bundle;
pub mod cache;
pub mod client;
pub mod config;
pub mod error;
pub mod format;
pub mod types;
pub mod prelude;
pub mod runtime;


// Re-exports for convenience
pub use bundle::PawBundle;
pub use cache::CacheManager;
pub use client::{PawClient, RawPawClient, CompileRequest, CompileRequestBuilder};
pub use config::PawConfig;
pub use error::{Error, Result};
pub use format::{
    ExamplePair, GenerationConfig, LoRAConfig, PawFileMeta, PawFormatReader, PawFormatWriter,
    TensorData,
};
pub use runtime::{PawFnTrait, PawRuntimeOptions};
pub use types::*;

/// High-level convenience: compile a spec on the PAW server.
///
/// Equivalent to `paw.compile(spec, compiler, slug, ...)` in the Python SDK.
pub async fn compile(
    spec: &str,
    config: &PawConfig,
    compiler: Option<&str>,
    slug: Option<&str>,
) -> Result<Program> {
    let client = PawClient::new(config);
    let mut b = CompileRequest::builder().spec(spec);
    if let Some(c) = compiler {
        b = b.compiler(c);
    }
    if let Some(s) = slug {
        b = b.slug(s);
    }
    client.compile(b.build()?).await
}

/// High-level convenience: compile and immediately download a .paw bundle.
///
/// Equivalent to `paw.compile_and_load(spec, ...)` in the Python SDK.
/// Returns the bundle and the local program directory.
pub async fn compile_and_download(
    spec: &str,
    config: &PawConfig,
    compiler: Option<&str>,
    slug: Option<&str>,
) -> Result<(Program, PawBundle)> {
    let client = PawClient::new(config);
    let mut b = CompileRequest::builder().spec(spec);
    if let Some(c) = compiler {
        b = b.compiler(c);
    }
    if let Some(s) = slug {
        b = b.slug(s);
    }
    let program = client.compile(b.build()?).await?;
    let dir = client.download_paw(&program.id).await?;
    let bundle = PawBundle::load_from_dir(&dir)?;
    Ok((program, bundle))
}

/// Resolve a slug to a program ID, checking local cache first.
pub async fn resolve(
    slug: &str,
    config: &PawConfig,
    offline: bool,
) -> Result<String> {
    let cache = CacheManager::new(config);

    if offline {
        return cache
            .get_cached_slug(slug)
            .ok_or_else(|| Error::Other(format!("No cached version of '{slug}'")));
    }

    if let Some(id) = cache.get_cached_slug(slug) {
        return Ok(id);
    }

    let client = PawClient::new(config);
    let program_id = client.resolve_slug(slug).await?;
    cache.save_slug_mapping(slug, &program_id);
    Ok(program_id)
}

/// Login: save an API key to the config file.
pub fn login(key: &str) -> std::io::Result<()> {
    config::set_api_key(key)
}

/// Clear the entire local PAW cache (base models, programs, runtimes).
pub fn clear_cache(config: &PawConfig) -> Result<()> {
    let cache = CacheManager::new(config);
    let size = cache.total_size().unwrap_or(0);
    cache.clear()?;
    println!("Cleared {:.1} MB from cache", size as f64 / 1_048_576.0);
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::CompileRequest;
    use crate::types::ProgramId;

    // ── ProgramId ─────────────────────────────────────────────────────

    #[test]
    fn test_program_id_valid_hex() {
        let id = ProgramId::new("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4").unwrap();
        assert_eq!(id.as_str(), "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
    }

    #[test]
    fn test_program_id_too_short() {
        let err = ProgramId::new("abc").unwrap_err();
        assert!(err.to_string().contains("Invalid program ID"));
    }

    #[test]
    fn test_program_id_non_hex() {
        let err = ProgramId::new("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz").unwrap_err();
        assert!(err.to_string().contains("Invalid program ID"));
    }

    #[test]
    fn test_program_id_unchecked() {
        let id = ProgramId::new_unchecked("anything");
        assert_eq!(id.as_str(), "anything");
    }

    #[test]
    fn test_program_id_is_hash_id() {
        assert!(ProgramId::is_hash_id("a1b2c3d4e5f6a1b2"));
        assert!(!ProgramId::is_hash_id("short"));
        assert!(!ProgramId::is_hash_id("has-hyphen-here"));
        assert!(ProgramId::is_hash_id("ABCDEF0123456789abcdef0123456789"));
    }

    #[test]
    fn test_program_id_from_str() {
        use std::str::FromStr;
        let id = ProgramId::from_str("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4").unwrap();
        assert_eq!(id.as_str(), "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4");
    }

    #[test]
    fn test_program_id_into_string() {
        let id = ProgramId::new_unchecked("abc123");
        let s: String = id.into();
        assert_eq!(s, "abc123");
    }

    #[test]
    fn test_program_id_display() {
        let id = ProgramId::new_unchecked("test-id");
        assert_eq!(format!("{id}"), "test-id");
    }

    // ── Slug ──────────────────────────────────────────────────────────

    #[test]
    fn test_slug_new() {
        use crate::types::Slug;
        let s = Slug::new("my-program");
        assert_eq!(s.as_str(), "my-program");
    }

    #[test]
    fn test_slug_from_str() {
        use crate::types::Slug;
        use std::str::FromStr;
        let s = Slug::from_str("hello-world").unwrap();
        assert_eq!(s.as_str(), "hello-world");
    }

    // ── Program ───────────────────────────────────────────────────────

    fn make_program(id: &str, slug: Option<&str>, version: Option<u32>) -> Program {
        Program {
            id: id.into(),
            status: "ready".into(),
            slug: slug.map(String::from),
            compiler_snapshot: None,
            compiler_kind: None,
            pseudo_program_strategy: None,
            runtime_id: None,
            runtime_manifest_version: None,
            timings: None,
            error: None,
            version,
            version_action: None,
        }
    }

    #[test]
    fn test_program_display_label_with_slug() {
        let p = make_program("abc123", Some("my-slug"), Some(1));
        assert_eq!(p.display_label(), "my-slug");
    }

    #[test]
    fn test_program_display_label_with_slug_v2() {
        let p = make_program("abc123", Some("my-slug"), Some(2));
        assert_eq!(p.display_label(), "my-slug v2");
    }

    #[test]
    fn test_program_display_label_no_slug() {
        let p = make_program("abc123", None, None);
        assert_eq!(p.display_label(), "abc123");
    }

    // ── BaseModelInfo ─────────────────────────────────────────────────

    #[test]
    fn test_base_model_download_url_with_url() {
        let info = BaseModelInfo {
            provider: "hf".into(),
            repo: "org/model".into(),
            filename: "model.gguf".into(),
            url: Some("https://example.com/model.gguf".into()),
            sha256: None,
        };
        assert_eq!(info.download_url(), "https://example.com/model.gguf");
    }

    #[test]
    fn test_base_model_download_url_generated() {
        let info = BaseModelInfo {
            provider: "hf".into(),
            repo: "org/model".into(),
            filename: "model.gguf".into(),
            url: None,
            sha256: None,
        };
        assert_eq!(
            info.download_url(),
            "https://huggingface.co/org/model/resolve/main/model.gguf"
        );
    }

    // ── CompileRequestBuilder ─────────────────────────────────────────

    #[test]
    fn test_compile_request_builder_minimal() {
        let req = CompileRequest::builder()
            .spec("test spec")
            .build()
            .unwrap();
        assert_eq!(req.spec, "test spec");
        assert!(req.compiler.is_none());
        assert!(!req.public); // important: default is false
        assert!(!req.ephemeral);
    }

    #[test]
    fn test_compile_request_builder_public_explicit() {
        let req = CompileRequest::builder().spec("x").public(true).build().unwrap();
        assert!(req.public);
        let req = CompileRequest::builder().spec("x").public(false).build().unwrap();
        assert!(!req.public);
    }

    #[test]
    fn test_compile_request_builder_full() {
        let req = CompileRequest::builder()
            .spec("classify")
            .compiler("paw-4b-qwen3-0.6b")
            .name("my-program")
            .tags(vec!["tag1".into(), "tag2".into()])
            .public(false)
            .slug("my-classifier")
            .ephemeral(true)
            .build()
            .unwrap();
        assert_eq!(req.spec, "classify");
        assert_eq!(req.compiler.unwrap(), "paw-4b-qwen3-0.6b");
        assert_eq!(req.name.unwrap(), "my-program");
        assert_eq!(req.tags.unwrap(), vec!["tag1", "tag2"]);
        assert!(!req.public);
        assert_eq!(req.slug.unwrap(), "my-classifier");
        assert!(req.ephemeral);
    }

    #[test]
    fn test_compile_request_builder_missing_spec() {
        let err = CompileRequest::builder().build().unwrap_err();
        assert!(err.to_string().contains("spec is required"));
    }

    #[test]
    fn test_compile_request_builder_ephemeral_default() {
        let req = CompileRequest::builder().spec("x").build().unwrap();
        assert!(!req.ephemeral);
    }

    // ── PawConfig / PawConfigBuilder ──────────────────────────────────

    #[test]
    fn test_paw_config_defaults() {
        // Use builder with no overrides — falls back to env defaults
        let config = PawConfig::builder().build().unwrap();
        assert_eq!(config.n_ctx(), 2048);
        assert_eq!(config.n_gpu_layers(), -1);
        assert!(!config.verbose());
        assert!(!config.offline());
    }

    #[test]
    fn test_paw_config_builder_full() {
        let config = PawConfig::builder()
            .api_url("https://custom.example.com")
            .api_key("sk-test-key")
            .n_ctx(4096)
            .n_gpu_layers(0)
            .verbose(true)
            .offline(true)
            .build()
            .unwrap();
        assert_eq!(config.api_url(), "https://custom.example.com");
        assert_eq!(config.api_key().unwrap(), "sk-test-key");
        assert_eq!(config.n_ctx(), 4096);
        assert_eq!(config.n_gpu_layers(), 0);
        assert!(config.verbose());
        assert!(config.offline());
    }

    #[test]
    fn test_paw_config_builder_partial() {
        let config = PawConfig::builder()
            .api_key("key-only")
            .verbose(true)
            .build()
            .unwrap();
        assert_eq!(config.api_key().unwrap(), "key-only");
        assert!(config.verbose());
        // other fields fall back to defaults
        assert_eq!(config.api_url(), "https://programasweights.com");
        assert_eq!(config.n_ctx(), 2048);
    }

    #[test]
    fn test_paw_config_base_models_dir() {
        let config = PawConfig::from_env();
        let dir = config.base_models_dir();
        assert!(dir.to_string_lossy().contains("base_models"));
    }

    #[test]
    fn test_paw_config_programs_dir() {
        let config = PawConfig::from_env();
        let dir = config.programs_dir();
        assert!(dir.to_string_lossy().contains("programs"));
    }

    #[test]
    fn test_paw_config_slug_cache_path() {
        let config = PawConfig::from_env();
        let path = config.slug_cache_path();
        assert!(path.to_string_lossy().contains("slug_cache.json"));
    }

    // ── Cache Manager ─────────────────────────────────────────────────

    #[test]
    fn test_cache_manager_creates_dirs() {
        let tmp = std::env::temp_dir().join("paw_test_cache");
        let config = PawConfig::builder()
            .cache_dir(tmp.clone())
            .build()
            .unwrap();
        let _cache = CacheManager::new(&config);
        // Verify the cache dirs are computed correctly
        let dirs = [
            config.base_models_dir(),
            config.programs_dir(),
            config.runtimes_dir(),
        ];
        for d in &dirs {
            let s = d.to_string_lossy();
            assert!(s.contains("paw_test_cache"), "unexpected dir: {s}");
        }
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_known_models_interpreter_to_gguf() {
        assert_eq!(
            cache::known_models::interpreter_to_gguf("Qwen/Qwen3-0.6B"),
            Some((
                cache::known_models::QWEN3_0_6B_GGUF_REPO,
                cache::known_models::QWEN3_0_6B_GGUF_FILE,
            ))
        );
        assert_eq!(
            cache::known_models::interpreter_to_gguf("gpt2"),
            Some((cache::known_models::GPT2_GGUF_REPO, cache::known_models::GPT2_GGUF_FILE))
        );
        assert_eq!(cache::known_models::interpreter_to_gguf("unknown"), None);
    }

    // ── Error type ────────────────────────────────────────────────────

    #[test]
    fn test_error_display_http() {
        // Can't construct Http without a real reqwest error.
        // Test the error enum variant display via Other for sanity.
        let err: Error = Error::Other("something went wrong".into());
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn test_error_display_api() {
        let err = Error::Api { status: 404, message: "Not Found".into() };
        assert_eq!(err.to_string(), "API error: 404 — Not Found");
    }

    #[test]
    fn test_error_display_not_found() {
        let err = Error::NotFound("program xyz".into());
        assert_eq!(err.to_string(), "Not found: program xyz");
    }

    #[test]
    fn test_error_display_timeout() {
        let err = Error::Timeout(120);
        assert_eq!(err.to_string(), "Timeout: program assets not ready after 120s");
    }

    // ── Result alias ──────────────────────────────────────────────────

    #[test]
    fn test_result_alias_ok() {
        let r: Result<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn test_result_alias_err() {
        let r: Result<i32> = Err(Error::Other("fail".into()));
        assert!(r.is_err());
    }
}
