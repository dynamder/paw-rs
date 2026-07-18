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

/// Magic bytes for prefix KV cache files.
const MAGIC: &[u8; 10] = b"PAWKVCACHE";

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
    /// Returns `true` if the cache was successfully loaded.
    pub fn try_load(&mut self) -> Result<bool, Error> {
        if !self.path.exists() {
            return Ok(false);
        }

        let data = std::fs::read(&self.path).map_err(Error::Io)?;
        if data.len() < 14 {
            return Ok(false);
        }

        if &data[..10] != MAGIC {
            return Ok(false);
        }

        let n_layers = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
        if n_layers != self.num_layers {
            return Ok(false);
        }

        let elem_count = self.prefix_len * self.num_kv_heads * self.head_dim;
        let layer_bytes = elem_count * 4 * 2; // K + V, 4 bytes per f32
        let expected_size = 14 + n_layers * layer_bytes;
        if data.len() < expected_size {
            return Ok(false);
        }

        let mut kv_pairs = Vec::with_capacity(n_layers);
        let mut offset = 14usize;

        let shape = &[1, self.num_kv_heads, self.prefix_len, self.head_dim];

        for _ in 0..n_layers {
            let k_bytes = &data[offset..offset + elem_count * 4];
            let k_vals: Vec<f32> = k_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            offset += elem_count * 4;

            let v_bytes = &data[offset..offset + elem_count * 4];
            let v_vals: Vec<f32> = v_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            offset += elem_count * 4;

            let k_t = Tensor::from_vec(k_vals, shape, &self.device)
                .map_err(|e| Error::Other(format!("kv cache load k: {e}")))?;
            let v_t = Tensor::from_vec(v_vals, shape, &self.device)
                .map_err(|e| Error::Other(format!("kv cache load v: {e}")))?;
            kv_pairs.push((k_t, v_t));
        }

        self.cache = Some(kv_pairs);
        Ok(true)
    }

    /// Store the current KV cache to disk.
    pub fn save(&self, kv_pairs: &[(Tensor, Tensor)]) -> Result<(), Error> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(kv_pairs.len() as u32).to_le_bytes());

        for (k, v) in kv_pairs {
            let k_data: Vec<f32> = k
                .flatten_all()
                .map_err(|e| Error::Other(format!("kv cache flatten k: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| Error::Other(format!("kv cache to_vec1 k: {e}")))?;
            let v_data: Vec<f32> = v
                .flatten_all()
                .map_err(|e| Error::Other(format!("kv cache flatten v: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| Error::Other(format!("kv cache to_vec1 v: {e}")))?;
            for &val in &k_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
            for &val in &v_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        std::fs::write(&self.path, buf).map_err(Error::Io)?;
        Ok(())
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
