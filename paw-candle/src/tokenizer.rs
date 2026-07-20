use paw_core::{Error, PawBundle};
use std::path::Path;

/// Wraps a HuggingFace `tokenizers` tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_id: u32,
}

impl Tokenizer {
    /// Load a tokenizer from the program bundle directory.
    /// Always uses the HuggingFace `tokenizer.json` from the program directory.
    pub fn new(bundle: &PawBundle) -> Result<Self, Error> {
        Self::from_file(bundle.program_dir.join("tokenizer.json"))
    }

    /// Load a tokenizer from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Other(format!("Failed to load tokenizer: {e}")))?;

        // Use ByteLevel pre-tokenizer to match GGUF-embedded tokenizer behavior.
        // The HF tokenizer.json uses a complex regex Sequence, but the GGUF model
        // uses simple ByteLevel (like GPT-2). We must match the GGUF tokenizer.
        inner.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
        ));
        inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(
            true, true, false,
        )));

        // Get EOS token ID from the tokenizer
        let eos_token_id = inner
            .token_to_id("<|im_end|>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("</s>"))
            .unwrap_or(0);

        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    /// Encode text using the inner tokenizer's native encoding.
    /// Special tokens in the vocabulary are handled by the BPE model.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Other(format!("token encode: {e}")))?;
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
