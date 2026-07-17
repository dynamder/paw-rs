//! # PAW Core — ProgramAsWeights Rust SDK core
//!
//! This crate provides the core types, HTTP client, cache management, and
//! bundle parsing for the ProgramAsWeights (PAW) ecosystem.
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

pub mod bundle;
pub mod cache;
pub mod client;
pub mod config;
pub mod error;
pub mod format;
pub mod types;
pub mod prelude;

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
