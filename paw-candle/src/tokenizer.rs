use paw_core::{Error, PawBundle};
use std::path::Path;

/// Wraps a HuggingFace `tokenizers` tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_id: u32,
}

impl Tokenizer {
    /// Load a tokenizer from a PAW bundle directory.
    ///
    /// Looks for `tokenizer.json` inside the bundle directory.
    pub fn new(bundle: &PawBundle) -> Result<Self, Error> {
        let tokenizer_path = bundle.program_dir.join("tokenizer.json");
        Self::from_file(tokenizer_path)
    }

    /// Load a tokenizer from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Other(format!("Failed to load tokenizer: {e}")))?;

        let eos_id = inner
            .token_to_id("<|endoftext|>")
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .unwrap_or(0);

        Ok(Self {
            eos_token_id: eos_id,
            inner,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Other(format!("Tokenizer encode error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("Tokenizer decode error: {e}")))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }
}
