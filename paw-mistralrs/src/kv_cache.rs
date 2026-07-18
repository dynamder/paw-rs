use std::path::PathBuf;

use paw_core::Error;

/// Manages persistence of the prefix tokens to disk.
///
/// With mistralrs, we cannot directly save/load the internal KV cache
/// (it is managed by paged attention). Instead, we store the prefix
/// token IDs so the full prompt can be reconstructed on subsequent runs.
///
/// The current implementation stores the prefix text so we can detect
/// whether the prefix has changed between runs. Recomputing the prefix
/// is acceptable for the experimental phase.
#[derive(Clone)]
pub struct PrefixCache {
    path: PathBuf,
    prefix_text: String,
    prefix_tokens: Option<Vec<u32>>,
}

const MAGIC: &[u8; 12] = b"PAWMSCACHEv1";

impl PrefixCache {
    pub fn new(path: PathBuf, prefix_text: &str) -> Self {
        Self {
            path,
            prefix_text: prefix_text.to_string(),
            prefix_tokens: None,
        }
    }

    /// Try to load the prefix token cache from disk.
    /// Returns `true` if a valid cache was loaded.
    pub fn try_load(&mut self) -> Result<bool, Error> {
        if !self.path.exists() {
            return Ok(false);
        }

        let data = std::fs::read(&self.path).map_err(Error::Io)?;
        if data.len() < 12 {
            return Ok(false);
        }
        if &data[..12] != MAGIC {
            return Ok(false);
        }

        let text_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

        if 16 + text_len > data.len() {
            return Ok(false);
        }

        let stored_text = String::from_utf8_lossy(&data[16..16 + text_len]);

        if stored_text != self.prefix_text {
            return Ok(false);
        }

        let mut offset = 16 + text_len;
        let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let mut tokens = Vec::with_capacity(count);
        for _ in 0..count {
            if offset + 4 > data.len() {
                return Ok(false);
            }
            let t = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            tokens.push(t);
            offset += 4;
        }

        self.prefix_tokens = Some(tokens);
        Ok(true)
    }

    /// Store the prefix tokens to disk.
    pub fn save(&self, tokens: &[u32]) -> Result<(), Error> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);

        let text_bytes = self.prefix_text.as_bytes();
        buf.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(text_bytes);

        buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }

        std::fs::write(&self.path, buf).map_err(Error::Io)?;
        Ok(())
    }

    pub fn get_tokens(&self) -> Option<&[u32]> {
        self.prefix_tokens.as_deref()
    }

    pub fn set_tokens(&mut self, tokens: Vec<u32>) {
        self.prefix_tokens = Some(tokens);
    }
}
