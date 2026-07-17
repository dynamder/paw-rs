//! Prefix KV cache save/load.
//!
//! After the first inference run, the prefix KV cache is saved to disk
//! so subsequent runs can skip the ~2-3s prefix evaluation step.

#![allow(dead_code)]

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use paw_core::Error;

/// Manages persistence of the prefix KV cache to disk.
///
/// The KV cache for the prompt template prefix tokens is computed once
/// and saved to a file. On subsequent runs, it's restored from disk,
/// eliminating the cold-start prefix evaluation.
pub struct PrefixKvCache {
    path: PathBuf,
    num_layers: usize,
    head_dim: usize,
    num_kv_heads: usize,
    prefix_len: usize,
    device: Device,
    cache: Option<Vec<(Tensor, Tensor)>>,
}

impl PrefixKvCache {
    /// Create a new prefix KV cache manager.
    pub fn new(
        path: PathBuf,
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
        prefix_len: usize,
        device: &Device,
    ) -> Self {
        Self {
            path,
            num_layers,
            head_dim,
            num_kv_heads,
            prefix_len,
            device: device.clone(),
            cache: None,
        }
    }

    /// Try to load the prefix KV cache from disk.
    pub fn try_load(&mut self) -> Result<bool, Error> {
        if !self.path.exists() {
            return Ok(false);
        }
        Err(Error::Other("KV cache loading not yet implemented".into()))
    }

    /// Store the current KV cache to disk.
    pub fn save(&self, _kv_pairs: &[(Tensor, Tensor)]) -> Result<(), Error> {
        Err(Error::Other("KV cache saving not yet implemented".into()))
    }

    /// Get the cached prefix KV pairs, if loaded.
    pub fn get_cached(&self) -> Option<&[(Tensor, Tensor)]> {
        self.cache.as_deref()
    }

    /// Set the cached KV pairs.
    pub fn set_cache(&mut self, kv_pairs: Vec<(Tensor, Tensor)>) {
        self.cache = Some(kv_pairs);
    }
}
