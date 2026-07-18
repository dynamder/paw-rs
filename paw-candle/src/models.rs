//! Model implementations for each supported architecture.
//!
//! Each module provides a quantized GGUF model that implements the
//! [`QuantizedModel`] trait used by the inference runtime.

/// Qwen3-0.6B model.
pub mod qwen3;

/// GPT-2 model (simple decoder-only transformer).
pub mod gpt2;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::lora::GgufLoraAdapter;

/// Read a GGUF integer metadata field, supporting all numeric GGUF value types.
pub(crate) fn gguf_get(content: &gguf_file::Content, key: &str, default: usize) -> usize {
    use gguf_file::Value;
    content
        .metadata
        .get(key)
        .map(|v| match v {
            Value::U8(n) => *n as usize,
            Value::I8(n) => *n as usize,
            Value::U16(n) => *n as usize,
            Value::I16(n) => *n as usize,
            Value::U32(n) => *n as usize,
            Value::I32(n) => *n as usize,
            Value::U64(n) => *n as usize,
            Value::I64(n) => *n as usize,
            Value::String(s) => s.parse().unwrap_or(default),
            Value::F32(f) => *f as usize,
            Value::F64(f) => *f as usize,
            _ => default,
        })
        .unwrap_or(default)
}

/// Load all tensors from a GGUF file.
pub(crate) fn load_gguf_tensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<
    (
        gguf_file::Content,
        HashMap<String, candle_core::quantized::QTensor>,
    ),
    candle_core::Error,
> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut tensors = HashMap::new();
    for (name, _info) in &content.tensor_infos {
        let qtensor = content.tensor(&mut file, name, device)?;
        tensors.insert(name.clone(), qtensor);
    }
    Ok((content, tensors))
}

/// A quantized model that can perform autoregressive generation.
pub trait QuantizedModel: Send {
    /// Returns the model's device.
    fn device(&self) -> &Device;

    /// Returns the number of layers in the model.
    fn num_layers(&self) -> usize;

    /// Forward pass: process `input_ids` at the given position index.
    ///
    /// - `input_ids`: shape `[1, seq_len]`, the token IDs to process.
    /// - `position`: the starting position in the sequence (for KV cache indexing).
    ///
    /// Returns logits of shape `[1, seq_len, vocab_size]`.
    fn forward(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> std::result::Result<Tensor, candle_core::Error>;

    /// Get the embedding weight tensor (for tied embeddings / lm_head).
    fn embed_tokens(&self) -> &Tensor;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Hidden size.
    fn hidden_size(&self) -> usize;

    /// Head dimension.
    fn head_dim(&self) -> usize;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;

    /// Number of KV heads (for GQA / MQA).
    fn num_kv_heads(&self) -> usize;

    /// Get the EOS token ID.
    fn eos_token_id(&self) -> u32 {
        0
    }

    /// Get the model name/identifier.
    fn model_name(&self) -> &str;

    /// Attach LoRA adapter to the model (default: no-op).
    fn set_lora(&mut self, _adapter: &GgufLoraAdapter) -> usize {
        0
    }

    /// Load prefix KV cache into the model (default: no-op).
    fn set_prefix_cache(&mut self, _prefix: &[(Tensor, Tensor)]) {}

    /// Extract the first `prefix_len` tokens from each layer's KV cache.
    fn extract_prefix_cache(&self, _prefix_len: usize) -> Option<Vec<(Tensor, Tensor)>> {
        None
    }
}
