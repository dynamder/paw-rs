use paw_core::Error;
use std::path::PathBuf;

#[derive(Clone)]
pub struct PrefixCache {
    path: PathBuf,
    prefix_text: String,
    n_prefix_tokens: usize,
}

const MAGIC: &[u8; 12] = b"PAWLCACHEv01";

impl PrefixCache {
    pub fn new(path: PathBuf, prefix_text: &str) -> Self {
        Self {
            path,
            prefix_text: prefix_text.to_string(),
            n_prefix_tokens: 0,
        }
    }

    pub fn try_load(&mut self) -> Result<bool, Error> {
        if !self.path.exists() {
            return Ok(false);
        }
        let data = std::fs::read(&self.path).map_err(Error::Io)?;
        if data.len() < 16 || &data[..12] != MAGIC {
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
        self.n_prefix_tokens =
            u32::from_le_bytes(data[16 + text_len..20 + text_len].try_into().unwrap()) as usize;
        Ok(self.n_prefix_tokens > 0)
    }

    pub fn save(&self, n_tokens: usize) -> Result<(), Error> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        let text_bytes = self.prefix_text.as_bytes();
        buf.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(text_bytes);
        buf.extend_from_slice(&(n_tokens as u32).to_le_bytes());
        std::fs::write(&self.path, buf).map_err(Error::Io)?;
        Ok(())
    }

    pub fn n_prefix_tokens(&self) -> usize {
        self.n_prefix_tokens
    }
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}
